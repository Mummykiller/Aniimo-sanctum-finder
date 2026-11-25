"""
Multi-vector weighted matcher comparing starting-room-scaled door geometry
and hallway exit profiles against layouts.json.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from vision import DoorSignature


@dataclass(frozen=True)
class MatchResult:
    layout_id: int
    error: float
    confidence: float
    angle_diff_deg: float
    distance_rel_err: float
    exits_err: float


def shortest_angle_diff_deg(a: float, b: float) -> float:
    """Return shortest unsigned angular difference in degrees [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _score_against_entries(
    angle_deg: float,
    norm_dist: float,
    exits: tuple[float, float, float, float],
    entries: dict[str, Any],
    *,
    w_angle: float,
    w_distance: float,
    w_exits: float,
    penalty_scale: float = 1.0,
) -> list[MatchResult]:
    candidates: list[MatchResult] = []

    for key, ref in entries.items():
        if not isinstance(ref, dict):
            continue
        try:
            layout_id = int(key)
            ref_ang = float(ref["angle_deg"])
            ref_dist = float(ref["normalized_distance"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(ref_ang) and math.isfinite(ref_dist) and ref_dist > 0.0):
            continue

        ang_diff = shortest_angle_diff_deg(angle_deg, ref_ang)
        ang_err = ang_diff / 180.0
        dist_err = min(1.0, abs(norm_dist - ref_dist) / max(ref_dist, 1e-6))

        ref_exits = ref.get("exits")
        eff_w_angle, eff_w_dist, eff_w_exits = w_angle, w_distance, w_exits
        exits_err = 0.0
        has_valid_exits = False
        if isinstance(ref_exits, list) and len(ref_exits) == 4:
            try:
                parsed_exits = [float(b) for b in ref_exits]
                if all(math.isfinite(b) for b in parsed_exits):
                    exits_err = sum(abs(float(a) - b) for a, b in zip(exits, parsed_exits)) / 4.0
                    has_valid_exits = True
            except (TypeError, ValueError):
                pass
        if not has_valid_exits and w_exits > 0.0:
            total_w = max(w_angle + w_distance, 1e-6)
            eff_w_angle = w_angle / total_w
            eff_w_dist = w_distance / total_w
            eff_w_exits = 0.0

        weighted_err = (
            eff_w_angle * ang_err
            + eff_w_dist * dist_err
            + eff_w_exits * exits_err
        )
        confidence = max(0.0, (1.0 - weighted_err) * penalty_scale)

        candidates.append(
            MatchResult(
                layout_id=layout_id,
                error=weighted_err,
                confidence=confidence,
                angle_diff_deg=ang_diff,
                distance_rel_err=dist_err,
                exits_err=exits_err,
            )
        )

    candidates.sort(key=lambda c: c.error)
    return candidates


def match_layout(
    sig: DoorSignature,
    layouts_db: dict[str, Any],
    *,
    w_angle: float = 0.45,
    w_distance: float = 0.25,
    w_exits: float = 0.30,
    min_confidence: float = 0.65,
) -> tuple[MatchResult | None, list[MatchResult]]:
    """
    Compare a detected DoorSignature against all layouts in layouts_db.
    Includes a 180-degree swapped-door fail-safe if the primary pass fails.
    """
    entries = layouts_db.get("layouts")
    if not isinstance(entries, dict):
        return None, []
    if sig.started_from == "blue":
        # Player spawned in the blue side-entrance room; orange main room is in fog.
        # Reverse vector (blue -> orange becomes orange -> blue) and ignore unrevealed orange exits.
        total_w = max(w_angle + w_distance, 1e-6)
        candidates = _score_against_entries(
            sig.angle_deg + 180.0,
            sig.normalized_distance,
            sig.exits,
            entries,
            w_angle=w_angle / total_w,
            w_distance=w_distance / total_w,
            w_exits=0.0,
        )
        if candidates and candidates[0].confidence >= min_confidence:
            return candidates[0], candidates
        return None, candidates

    candidates = _score_against_entries(
        sig.angle_deg,
        sig.normalized_distance,
        sig.exits,
        entries,
        w_angle=w_angle,
        w_distance=w_distance,
        w_exits=w_exits,
    )
    if not candidates:
        return None, []

    if candidates[0].confidence >= min_confidence:
        return candidates[0], candidates

    # Fail-safe pass: swapped/180-degree reversed door orientation
    rev_exits = (sig.exits[2], sig.exits[3], sig.exits[0], sig.exits[1])
    rev_candidates = _score_against_entries(
        sig.angle_deg + 180.0,
        sig.normalized_distance,
        rev_exits,
        entries,
        w_angle=w_angle,
        w_distance=w_distance,
        w_exits=w_exits,
        penalty_scale=0.92,
    )
    if rev_candidates and rev_candidates[0].confidence >= min_confidence:
        return rev_candidates[0], rev_candidates

    return None, candidates
