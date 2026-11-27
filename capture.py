"""
Cross-platform primary-monitor screenshot capture.

- Windows & Linux X11: mss (sct.monitors[1] = primary monitor only)
- Linux Wayland (KDE Plasma / wlroots): detects $XDG_SESSION_TYPE == "wayland"
  or $WAYLAND_DISPLAY and uses spectacle (with kscreen-doctor primary monitor
  crop when multi-monitor) or grim fallback.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def is_wayland_session() -> bool:
    """Return True when running inside a Linux Wayland session."""
    if not sys.platform.startswith("linux"):
        return False
    session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    return session_type == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def _get_kde_primary_monitor_geometry() -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    """
    Query kscreen-doctor for enabled outputs and return:
      ((rel_x, rel_y, w, h), (desktop_w, desktop_h))
    in logical coordinates shifted so the virtual desktop origin is (0, 0).
    """
    kscreen = shutil.which("kscreen-doctor")
    if not kscreen:
        return None
    try:
        proc = subprocess.run(
            [kscreen, "-o"],
            check=True,
            timeout=3,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        clean = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
        blocks = re.split(r"(?=Output:\s+\d+)", clean)
        outputs: list[tuple[int, int, int, int, bool]] = []

        for blk in blocks:
            if not re.search(r"^\s*enabled\b", blk, re.MULTILINE):
                continue
            geom_m = re.search(r"Geometry:\s*(-?\d+),(-?\d+)\s+(\d+)x(\d+)", blk)
            if not geom_m:
                continue
            x, y, w, h = (
                int(geom_m.group(1)),
                int(geom_m.group(2)),
                int(geom_m.group(3)),
                int(geom_m.group(4)),
            )
            if w <= 0 or h <= 0:
                continue
            is_primary = bool(re.search(r"\bpriority\s+1\b", blk))
            outputs.append((x, y, w, h, is_primary))

        if not outputs:
            return None

        min_x = min(o[0] for o in outputs)
        min_y = min(o[1] for o in outputs)
        max_x = max(o[0] + o[2] for o in outputs)
        max_y = max(o[1] + o[3] for o in outputs)
        desktop_w = max_x - min_x
        desktop_h = max_y - min_y
        if desktop_w <= 0 or desktop_h <= 0:
            return None

        primary = next((o for o in outputs if o[4]), outputs[0])
        px, py, pw, ph, _ = primary
        return (px - min_x, py - min_y, pw, ph), (desktop_w, desktop_h)
    except Exception:
        return None


_WINDOWS_DPI_CONFIGURED = False


def _enable_windows_dpi_awareness() -> None:
    """Enable per-monitor DPI awareness (v2 -> v1 -> system) on Windows before mss queries displays."""
    global _WINDOWS_DPI_CONFIGURED
    if _WINDOWS_DPI_CONFIGURED or not sys.platform.startswith("win"):
        return
    _WINDOWS_DPI_CONFIGURED = True
    try:
        import ctypes

        # 1. Windows 10 1703+ / Windows 11: Per-Monitor Aware v2 (-4)
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            set_ctx = user32.SetProcessDpiAwarenessContext
            set_ctx.argtypes = [ctypes.c_void_p]
            set_ctx.restype = ctypes.c_bool
            if set_ctx(ctypes.c_void_p(-4)):
                return
        except Exception:
            pass

        # 2. Windows 8.1+: Per-Monitor Aware v1 (2)
        try:
            shcore = ctypes.WinDLL("shcore", use_last_error=True)
            set_aware = shcore.SetProcessDpiAwareness
            set_aware.argtypes = [ctypes.c_int]
            set_aware.restype = ctypes.c_long
            if set_aware(2) in (0, -2147024891):  # S_OK or E_ACCESSDENIED (already set)
                return
        except Exception:
            pass

        # 3. Legacy fallback
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.SetProcessDPIAware.argtypes = []
            user32.SetProcessDPIAware.restype = ctypes.c_bool
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def _capture_mss_primary() -> np.ndarray:
    """Capture the primary monitor via mss and return a BGR numpy array."""
    _enable_windows_dpi_awareness()
    import mss

    with mss.mss() as sct:
        if not sct.monitors:
            raise RuntimeError("No monitors reported by mss.")
        if len(sct.monitors) > 1:
            # Prefer mss >= 10.2 explicit is_primary flag, then OS primary origin (0, 0), else monitors[1].
            phys = sct.monitors[1:]
            monitor = next(
                (m for m in phys if m.get("is_primary") is True),
                next(
                    (m for m in phys if m.get("left") == 0 and m.get("top") == 0),
                    phys[0],
                ),
            )
        else:
            monitor = sct.monitors[0]
        raw = sct.grab(monitor)
        bgra = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def _capture_wayland_primary() -> np.ndarray:
    """Capture the primary monitor on Linux Wayland via spectacle or grim."""
    spectacle = shutil.which("spectacle")
    grim = shutil.which("grim")
    if not spectacle and not grim:
        raise RuntimeError(
            "Wayland session detected, but neither 'spectacle' (KDE) nor 'grim' is installed."
        )

    is_kde = "kde" in os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    use_spectacle_first = bool(spectacle) and (is_kde or not grim)
    kde_geom = _get_kde_primary_monitor_geometry() if (is_kde or spectacle) else None

    def _run_capture_tool(tool_name: str, cmd: list[str], out_path: Path) -> None:
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode(errors="replace").strip() if exc.stderr else str(exc)
            raise RuntimeError(f"{tool_name} failed: {err}") from exc
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"{tool_name} produced an empty file.")

    tmp_paths: list[Path] = []
    try:
        captured_path: Path | None = None
        last_err: Exception | None = None

        tools_to_try = []
        if use_spectacle_first and spectacle:
            tools_to_try.append("spectacle")
            if grim:
                tools_to_try.append("grim")
        else:
            if grim:
                tools_to_try.append("grim")
            if spectacle:
                tools_to_try.append("spectacle")

        for tool in tools_to_try:
            suffix = ".png" if tool == "spectacle" else ".ppm"
            fd, tmp_name = tempfile.mkstemp(prefix="aniimo_cap_", suffix=suffix)
            os.close(fd)
            tmp_path = Path(tmp_name)
            tmp_paths.append(tmp_path)
            if tool == "spectacle":
                flag = "-f" if kde_geom is not None else "-m"
                cmd = [spectacle, "-b", "-n", flag, "-o", str(tmp_path)]
            else:
                cmd = [grim, "-t", "ppm", str(tmp_path)]
            try:
                _run_capture_tool(tool, cmd, tmp_path)
                captured_path = tmp_path
                break
            except Exception as exc:
                last_err = exc

        if captured_path is None:
            raise RuntimeError(f"Wayland screenshot capture failed: {last_err}")

        bgr = cv2.imread(str(captured_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode Wayland screenshot image.")

        if kde_geom is not None:
            (px, py, pw, ph), (desktop_w, desktop_h) = kde_geom
            ih, iw = bgr.shape[:2]
            # Map logical coordinates to physical image pixels (handles HiDPI / fractional scaling).
            sx = iw / float(desktop_w)
            sy = ih / float(desktop_h)
            x0 = max(0, int(round(px * sx)))
            y0 = max(0, int(round(py * sy)))
            x1 = min(iw, int(round((px + pw) * sx)))
            y1 = min(ih, int(round((py + ph) * sy)))
            if x1 > x0 and y1 > y0 and (x0 > 0 or y0 > 0 or x1 < iw or y1 < ih):
                return np.ascontiguousarray(bgr[y0:y1, x0:x1])
        return bgr
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def capture_primary_monitor() -> np.ndarray:
    """Capture ONLY the primary monitor and return a BGR OpenCV numpy array."""
    if is_wayland_session():
        return _capture_wayland_primary()
    return _capture_mss_primary()
