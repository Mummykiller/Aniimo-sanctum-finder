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
import sys
import webbrowser
from pathlib import Path

import cv2

from capture import capture_primary_monitor
from matcher import match_layout
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
            os.nice(10)
        except Exception:
            pass
    elif sys.platform.startswith("win"):
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass


def main() -> int:
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
    args = parser.parse_args()

    layouts_db = (
        sync_layouts()
        if not args.no_sync
        else sync_layouts(remote_url="")
    )

    if args.test_image is not None:
        image_bgr = cv2.imread(str(args.test_image))
        if image_bgr is None:
            print(f"[error] Could not read test image: {args.test_image}", file=sys.stderr)
            return 1
        trim = 0.0
    else:
        image_bgr = capture_primary_monitor()
        trim = 0.05

    sig, overlay, info = detect_doors(image_bgr, trim_margin=trim, return_debug=args.calibrate)

    if args.calibrate:
        out_overlay = BASE_DIR / "calibrate_overlay.png"
        if overlay is not None:
            cv2.imwrite(str(out_overlay), overlay)
            print(f"[calibrate] Saved debug overlay -> {out_overlay}")
        print(f"[calibrate] Detection info: {info}")

    if sig is None:
        print(
            "[info] Both orange and blue Sanctum doors were not detected on screen "
            f"(orange={info['orange_found']}, blue={info['blue_found']}). Exiting safely."
        )
        return 0

    best, candidates = match_layout(sig, layouts_db, min_confidence=args.min_confidence)
    if candidates:
        preview = ", ".join(
            f"Map {c.layout_id}: {c.confidence:.3f} "
            f"(ang_err={c.angle_diff_deg:.1f}°, dist_err={c.distance_rel_err:.2f}, exit_err={c.exits_err:.2f})"
            for c in candidates[:3]
        )
        print(f"[match] Top candidates -> {preview}")

    if best is None:
        print(
            f"[info] No layout met confidence threshold ({args.min_confidence:.2f}). Exiting safely."
        )
        return 0

    target_url = get_url_for_layout(best.layout_id)
    if not target_url:
        print(f"[warn] No URL configured for layout {best.layout_id} in urls.py.")
        return 1

    print(f"[launch] Map {best.layout_id} (confidence={best.confidence:.3f}) -> {target_url}")
    if not args.dry_run:
        webbrowser.open(target_url, new=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
