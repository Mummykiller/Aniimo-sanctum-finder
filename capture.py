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

import numpy as np
from PIL import Image


def is_wayland_session() -> bool:
    """Return True when running inside a Linux Wayland session."""
    if not sys.platform.startswith("linux"):
        return False
    session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    return session_type == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def _get_kde_primary_monitor_rect() -> tuple[int, int, int, int] | None:
    """Query kscreen-doctor for the primary (priority 1) monitor geometry."""
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
        fallback_rect: tuple[int, int, int, int] | None = None

        for blk in blocks:
            if "enabled" not in blk:
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
            if fallback_rect is None:
                fallback_rect = (x, y, w, h)
            if re.search(r"priority\s+1\b", blk):
                return (x, y, w, h)
        return fallback_rect
    except Exception:
        return None


def _capture_mss_primary() -> np.ndarray:
    """Capture the primary monitor via mss and return a BGR numpy array."""
    import mss

    with mss.mss() as sct:
        # sct.monitors[0] is the virtual desktop; sct.monitors[1] is primary.
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        raw = sct.grab(monitor)
        rgb = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return np.asarray(rgb)[:, :, ::-1].copy()


def _capture_wayland_primary() -> np.ndarray:
    """Capture the primary monitor on Linux Wayland via spectacle or grim."""
    fd, tmp_name = tempfile.mkstemp(prefix="aniimo_cap_", suffix=".png")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        spectacle = shutil.which("spectacle")
        grim = shutil.which("grim")
        primary_rect = _get_kde_primary_monitor_rect()

        if spectacle:
            if primary_rect is not None:
                cmd = [spectacle, "-b", "-n", "-f", "-o", str(tmp_path)]
            else:
                cmd = [spectacle, "-b", "-n", "-m", "-o", str(tmp_path)]
            subprocess.run(
                cmd,
                check=True,
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        elif grim:
            subprocess.run(
                [grim, str(tmp_path)],
                check=True,
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            raise RuntimeError(
                "Wayland session detected, but neither 'spectacle' (KDE) nor 'grim' is installed."
            )

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError("Wayland screenshot command produced an empty file.")

        with Image.open(tmp_path) as img:
            rgb = img.convert("RGB")
            if primary_rect is not None:
                px, py, pw, ph = primary_rect
                iw, ih = rgb.size
                if px >= 0 and py >= 0 and px + pw <= iw and py + ph <= ih:
                    rgb = rgb.crop((px, py, px + pw, py + ph))
            return np.asarray(rgb)[:, :, ::-1].copy()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def capture_primary_monitor() -> np.ndarray:
    """Capture ONLY the primary monitor and return a BGR OpenCV numpy array."""
    if is_wayland_session():
        return _capture_wayland_primary()
    return _capture_mss_primary()
