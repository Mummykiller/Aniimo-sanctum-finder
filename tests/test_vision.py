"""Unit tests for vision.py door detection, room measurement, and fallbacks."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matcher
import sync
import vision

IMAGE_PNG = ROOT / "image.png"
IMAGES_DIR = ROOT / "Images"


def _make_synthetic_map(
    *,
    width: int = 1280,
    height: int = 720,
    orange_pos: tuple[int, int] | None = (420, 360),
    blue_pos: tuple[int, int] | None = (820, 360),
    yellow_pos: tuple[int, int] | None = None,
    room_half: int = 65,
) -> np.ndarray:
    """Build a synthetic Sanctum-like BGR frame with room floors and door icons."""
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :] = (0, 0, 22)  # dark fog background

    def draw_room(cx: int, cy: int) -> None:
        x0, x1 = max(0, cx - room_half), min(width, cx + room_half)
        y0, y1 = max(0, cy - room_half), min(height, cy + room_half)
        hsv[y0:y1, x0:x1] = (15, 25, 195)  # bright desaturated floor

    for pos in (orange_pos, blue_pos, yellow_pos):
        if pos is not None:
            draw_room(*pos)

    if orange_pos is not None:
        ox, oy = orange_pos
        cv2.circle(hsv, (ox, oy), 12, (16, 170, 220), -1)
    if blue_pos is not None:
        bx, by = blue_pos
        cv2.circle(hsv, (bx, by), 12, (104, 210, 240), -1)
    if yellow_pos is not None:
        yx, yy = yellow_pos
        cv2.circle(hsv, (yx, yy), 12, (28, 220, 245), -1)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestVisionEdgeCases(unittest.TestCase):
    def test_blank_dark_frame_returns_none_cleanly(self) -> None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        sig, overlay, info = vision.detect_doors(img, trim_margin=0.05, retry_untrimmed=True)
        self.assertIsNone(sig)
        self.assertIsNone(overlay)
        self.assertFalse(info["orange_found"])
        self.assertFalse(info["blue_found"])
        self.assertIsNone(info["used_yellow_fallback"])

    def test_tiny_frame_does_not_crash(self) -> None:
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        sig, overlay, info = vision.detect_doors(img, trim_margin=0.05, retry_untrimmed=True, return_debug=True)
        self.assertIsNone(sig)
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape, (10, 10, 3))

    def test_all_bright_roi_with_debug_overlay_no_flt_max_crash(self) -> None:
        bright = np.full((720, 1280, 3), 240, dtype=np.uint8)
        bright[200:225, 300:325] = (10, 120, 255)  # orange
        bright[450:475, 800:825] = (255, 130, 10)  # blue
        sig, overlay, _ = vision.detect_doors(bright, trim_margin=0.0, return_debug=True)
        self.assertIsNotNone(overlay)
        if sig is not None:
            self.assertTrue(math.isfinite(sig.room_size))
            self.assertLess(sig.room_size, 10000.0)

    def test_yellow_only_non_map_frame_exits_quietly(self) -> None:
        img = _make_synthetic_map(orange_pos=None, blue_pos=None, yellow_pos=(640, 360))
        sig, _, info = vision.detect_doors(img, trim_margin=0.05, retry_untrimmed=True)
        self.assertIsNone(sig)
        self.assertFalse(info["orange_found"])
        self.assertFalse(info["blue_found"])
        self.assertIsNone(info["used_yellow_fallback"])

    def test_yellow_fallback_replaces_missing_orange_door(self) -> None:
        img = _make_synthetic_map(orange_pos=None, blue_pos=(820, 360), yellow_pos=(420, 360))
        sig, overlay, info = vision.detect_doors(img, trim_margin=0.0, return_debug=True)
        self.assertIsNotNone(sig)
        self.assertIsNotNone(overlay)
        self.assertTrue(info["orange_found"])
        self.assertTrue(info["blue_found"])
        self.assertEqual(info["used_yellow_fallback"], "orange")

    def test_yellow_fallback_replaces_missing_blue_door(self) -> None:
        img = _make_synthetic_map(orange_pos=(420, 360), blue_pos=None, yellow_pos=(820, 360))
        sig, _, info = vision.detect_doors(img, trim_margin=0.0)
        self.assertIsNotNone(sig)
        self.assertTrue(info["orange_found"])
        self.assertTrue(info["blue_found"])
        self.assertEqual(info["used_yellow_fallback"], "blue")

    def test_border_touching_room_stays_centered(self) -> None:
        hsv = np.zeros((400, 400, 3), dtype=np.uint8)
        hsv[:, :] = (0, 0, 20)
        hsv[100:200, 0:100] = (15, 20, 200)  # 100x100 room touching x=0
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rx, ry, rsize, exits = vision.measure_starting_room(bgr, 50.0, 150.0, 20.0, hsv=hsv)
        self.assertAlmostEqual(rx, 50.0, delta=4.0)
        self.assertAlmostEqual(ry, 150.0, delta=4.0)
        self.assertAlmostEqual(rsize, 100.0, delta=6.0)
        self.assertEqual(len(exits), 4)

    def test_retry_untrimmed_recovers_both_doors_in_margin(self) -> None:
        # Place orange door at x=30 (inside 5% margin x0=64), blue door in center.
        img = _make_synthetic_map(orange_pos=(30, 360), blue_pos=(820, 360), room_half=55)
        sig_trim, _, _ = vision.detect_doors(img, trim_margin=0.05, retry_untrimmed=False)
        self.assertIsNone(sig_trim)
        sig_retry, _, info_retry = vision.detect_doors(img, trim_margin=0.05, retry_untrimmed=True)
        self.assertIsNotNone(sig_retry)
        self.assertEqual(info_retry["roi_size"], (1280, 720))

    def test_retry_untrimmed_preserves_single_door_in_margin(self) -> None:
        img = _make_synthetic_map(orange_pos=(30, 360), blue_pos=None, room_half=55)
        sig, overlay, info = vision.detect_doors(img, trim_margin=0.05, retry_untrimmed=True, return_debug=True)
        self.assertIsNone(sig)
        self.assertIsNotNone(overlay)
        self.assertTrue(info["orange_found"])
        self.assertFalse(info["blue_found"])
        self.assertEqual(info["roi_size"], (1280, 720))

    def test_downscale_if_needed_4k_and_small(self) -> None:
        img_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
        scaled = vision._downscale_if_needed(img_4k, 1440)
        self.assertEqual(scaled.shape, (810, 1440, 3))

        small = np.zeros((720, 1280, 3), dtype=np.uint8)
        same = vision._downscale_if_needed(small, 1440)
        self.assertIs(same, small)

        disabled = vision._downscale_if_needed(img_4k, 0)
        self.assertIs(disabled, img_4k)

    @unittest.skipUnless(IMAGE_PNG.exists(), "image.png not present")
    def test_sample_image_matches_map_40(self) -> None:
        bgr = cv2.imread(str(IMAGE_PNG))
        self.assertIsNotNone(bgr)
        sig, overlay, info = vision.detect_doors(bgr, trim_margin=0.0, return_debug=True)
        self.assertIsNotNone(sig)
        self.assertIsNotNone(overlay)
        self.assertTrue(info["orange_found"] and info["blue_found"])
        db = sync.sync_layouts(remote_url="")
        best, _ = matcher.match_layout(sig, db)
        self.assertIsNotNone(best)
        self.assertEqual(best.layout_id, 40)

    @unittest.skipUnless(IMAGES_DIR.exists(), "Local Images/ dataset not present")
    def test_all_reference_layouts_match_bundled_db_regression(self) -> None:
        import build_layouts

        db = sync.sync_layouts(remote_url="")
        numeric_imgs = sorted(
            (p for p in IMAGES_DIR.rglob("*.png") if p.stem.isdigit()),
            key=build_layouts._sort_key,
        )
        self.assertGreaterEqual(len(numeric_imgs), 23)

        main_tpl, side_tpl = build_layouts._obtain_door_templates(IMAGES_DIR, numeric_imgs)
        scaled_main = build_layouts._precompute_scaled_templates(main_tpl)
        scaled_side = build_layouts._precompute_scaled_templates(side_tpl)

        seen_ids: set[int] = set()
        for img_path in numeric_imgs:
            lid = int(img_path.stem)
            if lid in seen_ids:
                continue
            seen_ids.add(lid)
            bgr = cv2.imread(str(img_path))
            self.assertIsNotNone(bgr, f"Failed to read {img_path}")
            vm, pm, hm = build_layouts._match_door_multiscale(bgr, scaled_main)
            vs, ps, _ = build_layouts._match_door_multiscale(bgr, scaled_side)
            self.assertGreaterEqual(vm, 0.55, f"Main door template weak on {img_path}")
            self.assertGreaterEqual(vs, 0.55, f"Side door template weak on {img_path}")
            rx, ry, rsize, exits = vision.measure_starting_room(
                bgr, pm[0], pm[1], float(hm), use_wall_scan=True
            )
            dx, dy = ps[0] - rx, ps[1] - ry
            sig = vision.DoorSignature(
                normalized_distance=math.hypot(dx, dy) / max(rsize, 1e-6),
                angle_deg=math.degrees(math.atan2(dy, dx)),
                size_ratio=1.0,
                orange_centroid=pm,
                blue_centroid=ps,
                orange_area=400.0,
                blue_area=400.0,
                orange_size_px=(int(hm), int(hm)),
                blue_size_px=(int(hm), int(hm)),
                room_center=(rx, ry),
                room_size=rsize,
                exits=exits,
                started_from="orange",
            )
            best, _ = matcher.match_layout(sig, db)
            self.assertIsNotNone(best, f"No match for reference {img_path}")
            self.assertEqual(best.layout_id, lid, f"Wrong match for {img_path}: got {best.layout_id}")
            self.assertGreaterEqual(best.confidence, 0.99, f"Drift on {img_path}: {best.confidence}")

    @unittest.skipUnless(IMAGES_DIR.exists(), "Local Images/ dataset not present")
    def test_real_gameplay_screenshots_regression(self) -> None:
        db = sync.sync_layouts(remote_url="")
        expected = [
            (IMAGES_DIR / "hard" / "Screenshot_20261122_163021.png", 50),
            (IMAGES_DIR / "chaos" / "Screenshot_20261122_163041.png", 58),
        ]
        for path, expected_id in expected:
            if not path.exists():
                continue
            bgr = cv2.imread(str(path))
            self.assertIsNotNone(bgr)
            sig, _, info = vision.detect_doors(bgr, trim_margin=0.0)
            self.assertIsNotNone(sig, f"Failed to detect doors on {path}: {info}")
            best, _ = matcher.match_layout(sig, db)
            self.assertIsNotNone(best, f"No match on {path}")
            self.assertEqual(best.layout_id, expected_id, f"Regression on {path}: got {best.layout_id}")


if __name__ == "__main__":
    unittest.main()
