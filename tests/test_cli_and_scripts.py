"""End-to-end CLI flow tests for main.py, build_layouts.py, and shell scripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_layouts
import main as app_main
from matcher import MatchResult
from vision import DoorSignature


def _dummy_sig() -> DoorSignature:
    return DoorSignature(
        normalized_distance=2.0,
        angle_deg=0.0,
        size_ratio=1.0,
        orange_centroid=(100.0, 100.0),
        blue_centroid=(200.0, 100.0),
        orange_area=400.0,
        blue_area=400.0,
        orange_size_px=(20, 20),
        blue_size_px=(20, 20),
        room_center=(100.0, 100.0),
        room_size=50.0,
        exits=(1.0, 0.0, 0.0, 0.0),
        started_from="orange",
    )


class TestMainFlows(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> int:
        with mock.patch.object(sys, "argv", ["main.py", *argv]), \
             mock.patch("main._set_low_cpu_priority"):
            return app_main.main()

    def test_high_confidence_match_opens_silently_and_captures_before_sync(self) -> None:
        order: list[str] = []
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        best = MatchResult(40, 0.05, 0.95, 1.0, 0.02, 0.0)

        with mock.patch("main.capture_primary_monitor", side_effect=lambda: (order.append("capture"), fake_img)[1]), \
             mock.patch("main.detect_doors", return_value=(_dummy_sig(), None, {"orange_found": True, "blue_found": True, "used_yellow_fallback": None})), \
             mock.patch("main.sync_layouts", side_effect=lambda **kw: (order.append("sync"), {"version": 1, "layouts": {}})[1]), \
             mock.patch("main.match_layout", return_value=(best, [best])), \
             mock.patch("main.webbrowser.open", side_effect=lambda u, new=2: order.append(f"open:{u}")) as mock_open, \
             mock.patch("main.send_notification") as mock_notify:
            rc = self._run_main([])

        self.assertEqual(rc, 0)
        self.assertEqual(order[:2], ["capture", "sync"])
        mock_open.assert_called_once()
        mock_notify.assert_not_called()

    def test_sub_90_confidence_opens_browser_before_notifying(self) -> None:
        order: list[str] = []
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        best = MatchResult(40, 0.20, 0.80, 5.0, 0.05, 0.1)

        with mock.patch("main.capture_primary_monitor", return_value=fake_img), \
             mock.patch("main.detect_doors", return_value=(_dummy_sig(), None, {"orange_found": True, "blue_found": True, "used_yellow_fallback": None})), \
             mock.patch("main.sync_layouts", return_value={"version": 1, "layouts": {}}), \
             mock.patch("main.match_layout", return_value=(best, [best])), \
             mock.patch("main.webbrowser.open", side_effect=lambda u, new=2: order.append("open")), \
             mock.patch("main.send_notification", side_effect=lambda t, m: order.append("notify")):
            rc = self._run_main([])

        self.assertEqual(rc, 0)
        self.assertEqual(order, ["open", "notify"])

    def test_zero_doors_exits_quietly_without_sync_or_notify(self) -> None:
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch("main.capture_primary_monitor", return_value=fake_img), \
             mock.patch("main.detect_doors", return_value=(None, None, {"orange_found": False, "blue_found": False, "used_yellow_fallback": None})), \
             mock.patch("main.sync_layouts") as mock_sync, \
             mock.patch("main.webbrowser.open") as mock_open, \
             mock.patch("main.send_notification") as mock_notify:
            rc = self._run_main([])

        self.assertEqual(rc, 0)
        mock_sync.assert_not_called()
        mock_open.assert_not_called()
        mock_notify.assert_not_called()

    def test_single_door_sends_warning_and_skips_browser(self) -> None:
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch("main.capture_primary_monitor", return_value=fake_img), \
             mock.patch("main.detect_doors", return_value=(None, None, {"orange_found": True, "blue_found": False, "used_yellow_fallback": None})), \
             mock.patch("main.webbrowser.open") as mock_open, \
             mock.patch("main.send_notification") as mock_notify:
            rc = self._run_main([])

        self.assertEqual(rc, 0)
        mock_open.assert_not_called()
        mock_notify.assert_called_once()
        self.assertIn("blue door not visible", mock_notify.call_args[0][1])

    def test_capture_failure_returns_1_and_notifies(self) -> None:
        with mock.patch("main.capture_primary_monitor", side_effect=RuntimeError("no display")), \
             mock.patch("main.send_notification") as mock_notify:
            rc = self._run_main([])
        self.assertEqual(rc, 1)
        mock_notify.assert_called_once()

    def test_invalid_min_confidence_exits_2(self) -> None:
        for bad in ("80", "-0.1", "nan", "inf"):
            with self.assertRaises(SystemExit) as cm:
                self._run_main(["--min-confidence", bad])
            self.assertEqual(cm.exception.code, 2)

    def test_nonexistent_test_image_returns_1(self) -> None:
        rc = self._run_main(["--test-image", str(ROOT / "does_not_exist_xyz.png"), "--no-sync", "--no-notify"])
        self.assertEqual(rc, 1)


class TestBuildLayoutsAndScripts(unittest.TestCase):
    def test_sort_key_is_deterministic_across_tiers(self) -> None:
        p1 = Path("Images/hard/40.png")
        p2 = Path("Images/beginner/40.png")
        self.assertEqual(sorted([p1, p2], key=build_layouts._sort_key), [p2, p1])

    @unittest.skipUnless(shutil.which("bash") is not None, "bash not installed")
    def test_shell_scripts_pass_syntax_check(self) -> None:
        for script in ("install_linux.sh", "launch.sh"):
            res = subprocess.run(["bash", "-n", str(ROOT / script)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
