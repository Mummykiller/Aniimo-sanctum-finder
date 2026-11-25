"""Unit tests for matcher.py and urls.py."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matcher
import urls
from vision import DoorSignature


def _sig(
    *,
    angle_deg: float = 0.0,
    normalized_distance: float = 2.0,
    exits: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    started_from: str = "orange",
) -> DoorSignature:
    return DoorSignature(
        normalized_distance=normalized_distance,
        angle_deg=angle_deg,
        size_ratio=1.0,
        orange_centroid=(100.0, 100.0),
        blue_centroid=(200.0, 100.0),
        orange_area=400.0,
        blue_area=400.0,
        orange_size_px=(20, 20),
        blue_size_px=(20, 20),
        room_center=(100.0, 100.0),
        room_size=50.0,
        exits=exits,
        started_from=started_from,
    )


class TestMatcher(unittest.TestCase):
    def test_shortest_angle_diff_wraparound(self) -> None:
        self.assertAlmostEqual(matcher.shortest_angle_diff_deg(179.0, -179.0), 2.0)
        self.assertAlmostEqual(matcher.shortest_angle_diff_deg(-180.0, 180.0), 0.0)
        self.assertAlmostEqual(matcher.shortest_angle_diff_deg(0.0, 180.0), 180.0)
        self.assertAlmostEqual(matcher.shortest_angle_diff_deg(350.0, 10.0), 20.0)

    def test_exact_orange_room_match(self) -> None:
        db = {
            "version": 1,
            "layouts": {
                "40": {"angle_deg": 45.0, "normalized_distance": 3.5, "exits": [1.0, 0.0, 1.0, 0.0]},
                "50": {"angle_deg": -90.0, "normalized_distance": 2.0, "exits": [0.0, 1.0, 0.0, 1.0]},
            },
        }
        best, cands = matcher.match_layout(
            _sig(angle_deg=45.0, normalized_distance=3.5, exits=(1.0, 0.0, 1.0, 0.0)), db
        )
        self.assertIsNotNone(best)
        self.assertEqual(best.layout_id, 40)
        self.assertAlmostEqual(best.confidence, 1.0)
        self.assertEqual(len(cands), 2)

    def test_blue_room_start_reverses_angle_and_ignores_exits(self) -> None:
        db = {
            "version": 1,
            "layouts": {
                "42": {"angle_deg": 30.0, "normalized_distance": 2.5, "exits": [0.0, 1.0, 0.0, 0.0]},
            },
        }
        # Blue-start vector points blue->orange (-150 deg), opposite of +30 deg, with completely different exits.
        best, _ = matcher.match_layout(
            _sig(angle_deg=-150.0, normalized_distance=2.5, exits=(1.0, 0.0, 1.0, 1.0), started_from="blue"),
            db,
        )
        self.assertIsNotNone(best)
        self.assertEqual(best.layout_id, 42)
        self.assertAlmostEqual(best.confidence, 1.0)

    def test_swapped_door_180_failsafe_applies_penalty(self) -> None:
        db = {
            "version": 1,
            "layouts": {
                "31": {"angle_deg": 0.0, "normalized_distance": 2.0, "exits": [0.0, 0.0, 1.0, 0.0]},
            },
        }
        # Opposite angle (180 deg) and rotated exits (N->S) triggers the 0.92 fail-safe.
        best, _ = matcher.match_layout(
            _sig(angle_deg=180.0, normalized_distance=2.0, exits=(1.0, 0.0, 0.0, 0.0)),
            db,
            min_confidence=0.65,
        )
        self.assertIsNotNone(best)
        self.assertEqual(best.layout_id, 31)
        self.assertAlmostEqual(best.confidence, 0.92, places=3)

    def test_min_confidence_gate_rejects_poor_match(self) -> None:
        db = {
            "version": 1,
            "layouts": {
                "31": {"angle_deg": 0.0, "normalized_distance": 1.0, "exits": [1.0, 1.0, 1.0, 1.0]},
            },
        }
        best, cands = matcher.match_layout(
            _sig(angle_deg=90.0, normalized_distance=5.0, exits=(0.0, 0.0, 0.0, 0.0)),
            db,
            min_confidence=0.80,
        )
        self.assertIsNone(best)
        self.assertEqual(len(cands), 1)

    def test_malformed_layouts_db_resilience(self) -> None:
        self.assertEqual(matcher.match_layout(_sig(), {"layouts": None}), (None, []))
        self.assertEqual(matcher.match_layout(_sig(), {"layouts": []}), (None, []))
        bad_db = {
            "version": 1,
            "layouts": {
                "not_int": {"angle_deg": 0.0, "normalized_distance": 2.0},
                "1": "not-a-dict",
                "2": {"angle_deg": float("nan"), "normalized_distance": 2.0},
                "3": {"angle_deg": 0.0, "normalized_distance": 0.0},
                "4": {"angle_deg": 0.0, "normalized_distance": 2.0, "exits": [None, 0, 1, 0]},
                "5": {"angle_deg": 0.0, "normalized_distance": 2.0, "exits": [1.0, 0.0, 0.0, 0.0]},
            },
        }
        best, cands = matcher.match_layout(_sig(), bad_db)
        self.assertIsNotNone(best)
        self.assertEqual(len(cands), 2)

    def test_missing_exits_does_not_get_free_bonus(self) -> None:
        db = {
            "version": 1,
            "layouts": {
                "1": {"angle_deg": 0.0, "normalized_distance": 2.0},  # missing exits
                "2": {"angle_deg": 0.0, "normalized_distance": 2.0, "exits": [1.0, 0.0, 0.0, 0.0]},
            },
        }
        _, cands = matcher.match_layout(_sig(angle_deg=10.0, normalized_distance=2.2), db)
        self.assertEqual(cands[0].layout_id, 2)


class TestUrls(unittest.TestCase):
    def test_get_url_for_layout(self) -> None:
        self.assertEqual(urls.get_url_for_layout(40), "https://aniimotools.dev/map/lost-sanctum/#plan-40")
        self.assertEqual(urls.get_url_for_layout(99), "https://aniimotools.dev/map/lost-sanctum/#plan-99")
        self.assertEqual(urls.get_url_for_layout(0), "")
        self.assertEqual(urls.get_url_for_layout(-3), "")

    def test_bundled_layouts_match_known_ids(self) -> None:
        bundled = json.loads((ROOT / "layouts.json").read_text(encoding="utf-8"))
        bundled_ids = {int(k) for k in bundled["layouts"]}
        self.assertEqual(bundled_ids, set(urls.KNOWN_LAYOUT_IDS))


if __name__ == "__main__":
    unittest.main()
