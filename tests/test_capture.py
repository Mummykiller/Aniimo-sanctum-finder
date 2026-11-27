"""Unit tests for capture.py (Wayland KDE/grim, HiDPI/negative offsets, mss)."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import capture


class TestCapture(unittest.TestCase):
    def test_is_wayland_session(self) -> None:
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            self.assertTrue(capture.is_wayland_session())

        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True):
            self.assertTrue(capture.is_wayland_session())

        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}, clear=True):
            self.assertFalse(capture.is_wayland_session())

        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            self.assertFalse(capture.is_wayland_session())

    def test_kde_geometry_ansi_negative_offset_and_disabled_hdr(self) -> None:
        raw_out = (
            "\x1b[32mOutput: 1 DP-1\x1b[0m\n"
            "    enabled\n"
            "    priority 2\n"
            "    Geometry: -1920,0 1920x1080\n"
            "    Scale: 1\n"
            "Output: 2 DP-2\n"
            "    enabled\n"
            "    priority 1\n"
            "    Geometry: 0,0 2560x1440\n"
            "    Scale: 1.5\n"
            "Output: 3 HDMI-1\n"
            "    disabled\n"
            "    HDR: enabled\n"
            "    Geometry: 2560,0 1920x1080\n"
        )
        with mock.patch("capture.shutil.which", return_value="/usr/bin/kscreen-doctor"), \
             mock.patch("capture.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=raw_out)):
            geom = capture._get_kde_primary_monitor_geometry()
        self.assertEqual(geom, ((1920, 0, 2560, 1440), (4480, 1440)))

    def test_kde_geometry_missing_tool_returns_none(self) -> None:
        with mock.patch("capture.shutil.which", return_value=None):
            self.assertIsNone(capture._get_kde_primary_monitor_geometry())

    def test_wayland_capture_crops_primary_with_hidpi_and_negative_offset(self) -> None:
        # Secondary on left (-100, 0, 100x100) red; Primary on right (0, 0, 100x100) green.
        # Captured at 2x HiDPI scale -> physical image is 400x200 (left 200px red, right 200px green).
        desktop_img = np.zeros((200, 400, 3), dtype=np.uint8)
        desktop_img[:, :200] = (0, 0, 255)   # secondary (red BGR)
        desktop_img[:, 200:] = (0, 255, 0)   # primary (green BGR)

        def fake_run(cmd, **kwargs):
            out_path = Path(cmd[-1])
            cv2.imwrite(str(out_path), desktop_img)
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "KDE"}, clear=False), \
             mock.patch("capture.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "spectacle" else None), \
             mock.patch("capture._get_kde_primary_monitor_geometry", return_value=((100, 0, 100, 100), (200, 100))), \
             mock.patch("capture.subprocess.run", side_effect=fake_run):
            result = capture._capture_wayland_primary()

        self.assertEqual(result.shape, (200, 200, 3))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        self.assertTrue(np.all(result == np.array([0, 255, 0], dtype=np.uint8)))

    def test_wayland_spectacle_failure_falls_back_to_grim(self) -> None:
        sample = np.full((60, 80, 3), 77, dtype=np.uint8)
        calls: list[str] = []

        def fake_run(cmd, **kwargs):
            tool = Path(cmd[0]).name
            calls.append(tool)
            if tool == "spectacle":
                raise subprocess.CalledProcessError(1, cmd, stderr=b"no kwin dbus")
            cv2.imwrite(cmd[-1], sample)
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "KDE"}, clear=False), \
             mock.patch("capture.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name in ("spectacle", "grim") else None), \
             mock.patch("capture._get_kde_primary_monitor_geometry", return_value=None), \
             mock.patch("capture.subprocess.run", side_effect=fake_run):
            result = capture._capture_wayland_primary()

        self.assertEqual(calls, ["spectacle", "grim"])
        self.assertEqual(result.shape, (60, 80, 3))

    def test_wayland_no_tools_raises(self) -> None:
        with mock.patch("capture.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                capture._capture_wayland_primary()

    def test_mss_primary_selects_origin_monitor_and_converts_bgra(self) -> None:
        # 2x3 BGRA buffer: B=10, G=20, R=30, A=255
        bgra_arr = np.zeros((2, 3, 4), dtype=np.uint8)
        bgra_arr[:, :] = (10, 20, 30, 255)
        grabbed_monitors: list[dict] = []

        class FakeMSS:
            monitors = [
                {"left": -1920, "top": 0, "width": 3840, "height": 1080},
                {"left": -1920, "top": 0, "width": 1920, "height": 1080},
                {"left": 0, "top": 0, "width": 1920, "height": 1080},
            ]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def grab(self, mon):
                grabbed_monitors.append(mon)
                return SimpleNamespace(bgra=bgra_arr.tobytes(), width=3, height=2)

        fake_mss_mod = SimpleNamespace(mss=FakeMSS)
        with mock.patch.dict(sys.modules, {"mss": fake_mss_mod}):
            bgr = capture._capture_mss_primary()

        self.assertEqual(grabbed_monitors, [{"left": 0, "top": 0, "width": 1920, "height": 1080}])
        self.assertEqual(bgr.shape, (2, 3, 3))
        self.assertTrue(np.all(bgr == np.array([10, 20, 30], dtype=np.uint8)))

    def test_mss_empty_monitors_raises(self) -> None:
        class EmptyMSS:
            monitors: list[dict] = []

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.dict(sys.modules, {"mss": SimpleNamespace(mss=EmptyMSS)}):
            with self.assertRaises(RuntimeError):
                capture._capture_mss_primary()

    def test_mss_prefers_explicit_is_primary_flag(self) -> None:
        bgra_arr = np.zeros((2, 2, 4), dtype=np.uint8)
        grabbed: list[dict] = []

        class FakeMSSPrimary:
            monitors = [
                {"left": 0, "top": 0, "width": 3840, "height": 1080},
                {"left": 0, "top": 0, "width": 1920, "height": 1080, "is_primary": False},
                {"left": 1920, "top": 0, "width": 1920, "height": 1080, "is_primary": True},
            ]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def grab(self, mon):
                grabbed.append(mon)
                return SimpleNamespace(bgra=bgra_arr.tobytes(), width=2, height=2)

        with mock.patch.dict(sys.modules, {"mss": SimpleNamespace(mss=FakeMSSPrimary)}):
            capture._capture_mss_primary()

        self.assertEqual(grabbed[0]["left"], 1920)
        self.assertTrue(grabbed[0]["is_primary"])


if __name__ == "__main__":
    unittest.main()
