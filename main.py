#!/usr/bin/env python3
"""
Aniimo Sanctum Launcher - Adaptive Hotkey Launcher

Orchestrates:
  1. sync.sync_layouts()           -> Load/update layouts.json (offline-safe)
  2. capture.capture_primary_monitor() -> Capture primary monitor only
  3. vision.detect_doors()         -> Adaptive HSV orange/blue door geometry
  4. matcher.match_layout()        -> Weighted geometric match + confidence gate
  5. webbrowser.open(url)          -> Open matching URL in default browser
"""

from __future__ import annotations

import os

for _env_key, _env_val in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("OPENCV_OPENCL_RUNTIME", "disabled"),
):
    os.environ.setdefault(_env_key, _env_val)

import argparse
import ctypes
import math
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np

from capture import capture_primary_monitor
from matcher import match_layout
from notify import APP_NAME, send_notification
from sync import sync_layouts
from urls import get_url_for_layout
from vision import detect_doors

BASE_DIR = Path(__file__).resolve().parent


def _set_low_cpu_priority() -> None:
    """Keep OpenCV/BLAS single-threaded and lower OS priority to avoid fan spikes."""
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    if os.name == "posix":
        try:
            cur_nice = os.nice(0)
            if cur_nice < 10:
                os.nice(10 - cur_nice)
        except Exception:
            pass
    elif sys.platform.startswith("win"):
        try:
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.SetPriorityClass.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass


def _configure_utf8_stdio() -> None:
    """Ensure stdout/stderr work safely under pythonw.exe and non-UTF-8 Windows locales."""
    if sys.stdout is None or sys.stderr is None:
        log_path = BASE_DIR / "launcher.log"
        stream = None
        try:
            if log_path.exists() and log_path.stat().st_size > 1_048_576:
                try:
                    log_path.replace(BASE_DIR / "launcher.log.1")
                except OSError:
                    pass
            stream = log_path.open("a", encoding="utf-8", errors="replace", buffering=1)
        except OSError:
            stream = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream
        return

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _read_image_unicode(path: Path) -> np.ndarray | None:
    """Read an image via Python bytes + cv2.imdecode so non-ASCII Windows paths work."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _write_image_unicode(path: Path, image: np.ndarray) -> bool:
    """Write an image via cv2.imencode + Path.write_bytes so non-ASCII Windows paths work."""
    try:
        ext = path.suffix or ".png"
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            return False
        path.write_bytes(encoded.tobytes())
        return True
    except OSError:
        return False


def main() -> int:
    _configure_utf8_stdio()
    _set_low_cpu_priority()
    parser = argparse.ArgumentParser(description="Aniimo Sanctum Adaptive Hotkey Launcher")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Save calibrate_overlay.png with detected contours and print HSV/geometry values",
    )
    parser.add_argument(
        "--test-image",
        type=Path,
        default=None,
        help="Run detection and matching against a local image file instead of capturing screen",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip checking remote GitHub layouts.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched layout and URL without opening the browser",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum match confidence [0..1] required to open a URL (default: 0.65)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable desktop notifications for low-confidence or partial-door captures",
    )
    args = parser.parse_args()
    if not (math.isfinite(args.min_confidence) and 0.0 <= args.min_confidence <= 1.0):
        parser.error("--min-confidence must be a finite number between 0.0 and 1.0")

    def maybe_notify(message: str) -> None:
        print(f"[notify] {message}")
        if not args.dry_run and not args.no_notify:
            send_notification(APP_NAME, message)

    # Capture screen BEFORE remote sync so slow/offline DNS never delays capturing the map.
    if args.test_image is not None:
        image_bgr = _read_image_unicode(args.test_image)
        if image_bgr is None:
            print(f"[error] Could not read test image: {args.test_image}", file=sys.stderr)
            return 1
        trim = 0.0
    else:
        try:
            image_bgr = capture_primary_monitor()
        except Exception as exc:
            print(f"[error] Screen capture failed: {exc}", file=sys.stderr)
            maybe_notify(f"Screen capture failed: {exc}")
            return 1
        trim = 0.05

    sig, overlay, info = detect_doors(
        image_bgr,
        trim_margin=trim,
        return_debug=args.calibrate,
        retry_untrimmed=(trim > 0.0),
    )

    if args.calibrate:
        out_overlay = BASE_DIR / "calibrate_overlay.png"
        if overlay is not None:
            if _write_image_unicode(out_overlay, overlay):
                print(f"[calibrate] Saved debug overlay -> {out_overlay}")
            else:
                print(f"[warn] Failed to write debug overlay -> {out_overlay}", file=sys.stderr)
        print(f"[calibrate] Detection info: {info}")

    if sig is None:
        if (
            sys.platform.startswith("win")
            and args.test_image is None
            and not info["orange_found"]
            and not info["blue_found"]
            and image_bgr.size > 0
            and int(image_bgr.max()) <= 2
        ):
            print("[warn] Captured an all-black screen on Windows (Exclusive Fullscreen?).")
            maybe_notify(
                "Captured a black screen. Switch Aniimo from Exclusive Fullscreen "
                "to Borderless Windowed mode and try again."
            )
            return 0
        print(
            "[info] Both orange and blue Sanctum doors were not detected on screen "
            f"(orange={info['orange_found']}, blue={info['blue_found']}). Exiting safely."
        )
        if info["orange_found"] != info["blue_found"]:
            missing = "blue" if info["orange_found"] else "orange"
            maybe_notify(
                f"Only one Sanctum door was detected ({missing} door not visible). "
                "Zoom out or center the map so both doors are clearly visible."
            )
        return 0

    layouts_db = (
        sync_layouts()
        if not args.no_sync
        else sync_layouts(remote_url="")
    )

    best, candidates = match_layout(sig, layouts_db, min_confidence=args.min_confidence)
    if candidates:
        preview = ", ".join(
            f"Map {c.layout_id}: {c.confidence:.3f} "
            f"(ang_err={c.angle_diff_deg:.1f}°, dist_err={c.distance_rel_err:.2f}, exit_err={c.exits_err:.2f})"
            for c in candidates[:3]
        )
        print(f"[match] Top candidates -> {preview}")

    if best is None:
        top_conf = candidates[0].confidence if candidates else 0.0
        print(
            f"[info] No layout met confidence threshold ({args.min_confidence:.2f}). Exiting safely."
        )
        if info.get("used_yellow_fallback"):
            maybe_notify(
                f"Low match confidence ({top_conf:.0%}) using player arrow fallback. "
                "Move off the door or zoom out so both doors are clearly visible."
            )
        else:
            maybe_notify(
                f"Low match confidence ({top_conf:.0%}). "
                "Open the full Sanctum map with both doors clearly visible and try again."
            )
        return 0

    target_url = get_url_for_layout(best.layout_id)
    if not target_url:
        print(f"[warn] No URL configured for layout {best.layout_id} in urls.py.")
        return 1

    print(f"[launch] Map {best.layout_id} (confidence={best.confidence:.3f}) -> {target_url}")
    if not args.dry_run:
        webbrowser.open(target_url, new=2)

    if info.get("used_yellow_fallback"):
        door_side = info["used_yellow_fallback"]
        maybe_notify(
            f"Matched Map {best.layout_id} ({best.confidence:.0%}) — might not be accurate, "
            f"using player arrow as {door_side} door fallback."
        )
    elif best.confidence < 0.90:
        maybe_notify(
            f"Matched Map {best.layout_id} ({best.confidence:.0%} confidence, below 90%). "
            "Zoom out or ensure both doors and room exits are clearly visible."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
