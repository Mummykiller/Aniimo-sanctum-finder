"""
Adaptive HSV color-masking & starting-room scale-invariant door geometry detector.

Extracts resolution- and zoom-invariant features anchored to the square starting room:
  1. Door-to-door geometry scaled by starting room width (normalized_distance, angle_deg)
  2. Starting-room 4-way hallway exit profile [N, E, S, W] sampled outside the 4 walls
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Default OpenCV HSV ranges (H: 0..179, S: 0..255, V: 0..255)
DEFAULT_HSV_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # Orange/amber crowned main door (H <= 21 avoids yellow player arrow; S <= 210 avoids bright yellow eggs)
    "orange": ((12, 65, 130), (21, 210, 255)),
    # Fallback when player's bright yellow arrow stands directly on a door
    "yellow_fallback": ((22, 120, 180), (35, 255, 255)),
    # Blue side entrance door
    "blue": ((98, 70, 120), (128, 255, 255)),
}

# Pre-allocated kernels and HSV threshold bounds (avoid per-call allocations)
_KERNEL_CLOSE_3X3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_FLOOR_LO = np.array((0, 0, 125), dtype=np.uint8)
_FLOOR_HI = np.array((179, 65, 255), dtype=np.uint8)
_ROOM_FLOOR_LO = np.array((0, 0, 118), dtype=np.uint8)
_ROOM_FLOOR_HI = np.array((179, 62, 255), dtype=np.uint8)


@dataclass(frozen=True)
class DoorSignature:
    normalized_distance: float
    angle_deg: float
    size_ratio: float
    orange_centroid: tuple[float, float]
    blue_centroid: tuple[float, float]
    orange_area: float
    blue_area: float
    orange_size_px: tuple[int, int]
    blue_size_px: tuple[int, int]
    room_center: tuple[float, float]
    room_size: float
    exits: tuple[float, float, float, float]
    started_from: str = "orange"

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_distance": round(self.normalized_distance, 4),
            "angle_deg": round(self.angle_deg, 3),
            "size_ratio": round(self.size_ratio, 4),
            "exits": [round(v, 3) for v in self.exits],
            "started_from": self.started_from,
        }


def crop_safe_margin(image_bgr: np.ndarray, margin: float = 0.10) -> np.ndarray:
    """Trim outer screen edges (taskbars/window borders) while keeping the map area."""
    if margin <= 0.0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    x0 = int(round(w * margin))
    x1 = int(round(w * (1.0 - margin)))
    y0 = int(round(h * margin))
    y1 = int(round(h * (1.0 - margin)))
    if x1 <= x0 or y1 <= y0:
        return image_bgr
    return image_bgr[y0:y1, x0:x1]


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Apply gentle morphological close to bridge door gaps without eroding small doors."""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_CLOSE_3X3, iterations=1)


def _floor_mask_from_hsv(hsv: np.ndarray) -> np.ndarray:
    """Isolate light-gray/tan walkable room & corridor floors from dark walls/fog."""
    return cv2.inRange(hsv, _FLOOR_LO, _FLOOR_HI)


def _room_floor_mask(hsv_roi: np.ndarray) -> np.ndarray:
    """Isolate walkable room floor pixels (V >= 118, S <= 62) in an HSV ROI."""
    return cv2.inRange(hsv_roi, _ROOM_FLOOR_LO, _ROOM_FLOOR_HI)


def _room_bright_mask(hsv_roi: np.ndarray) -> np.ndarray:
    """Isolate walkable floor + bright non-blue room features in an HSV ROI."""
    h_ch, s_ch, v_ch = cv2.split(hsv_roi)
    floor = cv2.bitwise_and(
        cv2.compare(v_ch, 118, cv2.CMP_GE),
        cv2.compare(s_ch, 62, cv2.CMP_LE),
    )
    non_blue = cv2.bitwise_or(
        cv2.compare(h_ch, 90, cv2.CMP_LT),
        cv2.compare(h_ch, 140, cv2.CMP_GT),
    )
    bright = cv2.bitwise_and(cv2.compare(v_ch, 135, cv2.CMP_GE), non_blue)
    cv2.bitwise_or(floor, bright, dst=bright)
    return bright


def _mask_fraction(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Return fraction of non-zero pixels in mask[y0:y1, x0:x1] (works for 0/1 or 0/255)."""
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return cv2.countNonZero(mask[y0:y1, x0:x1]) / float((x1 - x0) * (y1 - y0))


def _has_floor_surround(floor_mask: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Return fraction of walkable floor pixels around (cx, cy) to reject off-map UI icons."""
    h, w = floor_mask.shape
    r = max(8, int(round(radius)))
    x0 = max(0, int(round(cx)) - r)
    x1 = min(w, int(round(cx)) + r + 1)
    y0 = max(0, int(round(cy)) - r)
    y1 = min(h, int(round(cy)) + r + 1)
    return _mask_fraction(floor_mask, x0, y0, x1, y1)


def _find_best_door_contour(
    mask: np.ndarray,
    floor_mask: np.ndarray,
    roi_w: int,
    roi_h: int,
    min_area: float,
    max_area: float,
    exclude_center: tuple[float, float] | None = None,
    min_sep_px: float = 20.0,
    min_solidity: float = 0.35,
) -> tuple[np.ndarray, tuple[float, float], float, tuple[int, int]] | None:
    """
    Find the most door-like contour by filtering pixel width/height, aspect ratio,
    solidity, and walkable-floor surroundings (doors sit inside rooms, not in empty space).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_w = max(6, int(round(roi_w * 0.004)))
    max_w = max(90, int(round(roi_w * 0.12)))
    min_h = max(8, int(round(roi_h * 0.008)))
    max_h = max(110, int(round(roi_h * 0.14)))

    best: tuple[np.ndarray, tuple[float, float], float, tuple[int, int]] | None = None
    best_score = -1.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bw < min_w or bw > max_w or bh < min_h or bh > max_h:
            continue

        aspect = bw / float(bh)
        if aspect < 0.40 or aspect > 1.85:
            continue

        solidity = area / float(bw * bh)
        if solidity < min_solidity:
            continue

        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

        if exclude_center is not None:
            if math.hypot(cx - exclude_center[0], cy - exclude_center[1]) < min_sep_px:
                continue

        floor_ratio = _has_floor_surround(floor_mask, cx, cy, radius=max(bw, bh) * 1.3)
        shape_score = max(0.2, 1.0 - abs(aspect - 0.75))
        floor_boost = 0.25 + floor_ratio
        score = area * shape_score * solidity * floor_boost
        if score > best_score:
            best_score = score
            best = (cnt, (cx, cy), area, (bw, bh))

    return best


def _exit_probe_boxes(
    rx: float, ry: float, half: float, img_w: int, img_h: int
) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) bounding boxes for N, E, S, W exit probes outside the flat room walls."""
    boxes: list[tuple[int, int, int, int]] = []
    for ax, sgn in (("y", -1), ("x", +1), ("y", +1), ("x", -1)):
        if ax == "y":
            ya = int(round(ry + sgn * 1.15 * half))
            yb = int(round(ry + sgn * 1.50 * half))
            y0, y1 = max(0, min(ya, yb)), min(img_h, max(ya, yb))
            x0 = max(0, int(round(rx - 0.18 * half)))
            x1 = min(img_w, int(round(rx + 0.18 * half)))
        else:
            xa = int(round(rx + sgn * 1.15 * half))
            xb = int(round(rx + sgn * 1.50 * half))
            x0, x1 = max(0, min(xa, xb)), min(img_w, max(xa, xb))
            y0 = max(0, int(round(ry - 0.18 * half)))
            y1 = min(img_h, int(round(ry + 0.18 * half)))
        boxes.append((x0, y0, x1, y1))
    return boxes


def _corner_probe_boxes(
    rx: float, ry: float, half: float, img_w: int, img_h: int
) -> list[tuple[int, int, int, int]]:
    """Return 4 square probe boxes (NW, NE, SE, SW) at the outer diagonal corners."""
    r = max(3, int(round(0.14 * half)))
    boxes: list[tuple[int, int, int, int]] = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        cx = int(round(rx + sx * 1.25 * half))
        cy = int(round(ry + sy * 1.25 * half))
        boxes.append((max(0, cx - r), max(0, cy - r), min(img_w, cx + r + 1), min(img_h, cy + r + 1)))
    return boxes


def _validate_room_corners(
    floor: np.ndarray, rx: float, ry: float, half: float
) -> tuple[bool, tuple[float, float, float, float]]:
    """Verify all 4 outer corners are dark walls (< 0.20 floor fraction)."""
    h, w = floor.shape
    vals: list[float] = []
    for x0, y0, x1, y1 in _corner_probe_boxes(rx, ry, half, w, h):
        if x1 <= x0 or y1 <= y0:
            vals.append(1.0)
        else:
            vals.append(_mask_fraction(floor, x0, y0, x1, y1))
    return all(v < 0.20 for v in vals), (vals[0], vals[1], vals[2], vals[3])


def _validate_room_corners_from_hsv(
    hsv: np.ndarray, rx: float, ry: float, half: float
) -> tuple[bool, tuple[float, float, float, float]]:
    """Verify all 4 outer corners are dark walls (< 0.20 floor fraction) directly on tiny HSV probes."""
    h, w = hsv.shape[:2]
    vals: list[float] = []
    for x0, y0, x1, y1 in _corner_probe_boxes(rx, ry, half, w, h):
        if x1 <= x0 or y1 <= y0:
            vals.append(1.0)
        else:
            probe_floor = _room_floor_mask(hsv[y0:y1, x0:x1])
            vals.append(cv2.countNonZero(probe_floor) / float((x1 - x0) * (y1 - y0)))
    return all(v < 0.20 for v in vals), (vals[0], vals[1], vals[2], vals[3])


def _snap_flat_walls_from_solid(
    solid: np.ndarray, rx_local: float, ry_local: float, half: float
) -> tuple[float, float, float]:
    """
    Refine DIST_C peak & half-size by snapping to the flat outer walls of `solid`
    along off-center strips (skipping central hallways and notched corners).
    """
    sh, sw = solid.shape
    step_start = max(2, int(0.5 * half))
    step_end = max(step_start + 4, int(1.6 * half))
    fracs = (-0.50, -0.40, 0.40, 0.50)

    def scan_side(axis: str, sign: int) -> float:
        hits: list[float] = []
        base = rx_local if axis == "x" else ry_local
        base_i = int(round(base))
        limit = sw if axis == "x" else sh
        other_base = ry_local if axis == "x" else rx_local
        other_limit = sh if axis == "x" else sw

        for f in fracs:
            other_i = int(round(other_base + f * half))
            if other_i < 0 or other_i >= other_limit:
                continue
            if sign > 0:
                lo = base_i + step_start
                hi = min(limit, base_i + step_end)
                if lo < 0 or lo >= hi:
                    continue
                line = solid[other_i, lo:hi] if axis == "x" else solid[lo:hi, other_i]
            else:
                hi = base_i - step_start
                lo = max(0, base_i - step_end + 1)
                if hi < 0 or hi >= limit or lo > hi:
                    continue
                line = (
                    solid[other_i, lo : hi + 1][::-1]
                    if axis == "x"
                    else solid[lo : hi + 1, other_i][::-1]
                )
            zeros = np.flatnonzero(line == 0)
            if zeros.size:
                hits.append(base + sign * (step_start + int(zeros[0])))
        return float(np.median(hits)) if hits else base + sign * half

    x_l = scan_side("x", -1)
    x_r = scan_side("x", +1)
    y_t = scan_side("y", -1)
    y_b = scan_side("y", +1)

    w_span = max(16.0, x_r - x_l)
    h_span = max(16.0, y_b - y_t)
    if abs(w_span - h_span) / max(w_span, h_span) <= 0.20:
        room_size = (w_span + h_span) / 2.0
    else:
        room_size = min(w_span, h_span)
    return (x_l + x_r) / 2.0, (y_t + y_b) / 2.0, max(16.0, room_size)


def _first_wall_hit(is_floor: np.ndarray, step_start: int, run_len: int) -> int | None:
    """Return the first step index where `run_len` consecutive non-floor pixels follow floor."""
    true_idx = np.flatnonzero(is_floor)
    if true_idx.size == 0:
        return None
    first_true = int(true_idx[0])
    dark = ~is_floor[first_true:]
    if dark.size < run_len:
        return None
    if run_len == 5:
        hit = dark[:-4] & dark[1:-3] & dark[2:-2] & dark[3:-1] & dark[4:]
    elif run_len == 4:
        hit = dark[:-3] & dark[1:-2] & dark[2:-1] & dark[3:]
    else:
        hit = np.ones(dark.size - run_len + 1, dtype=bool)
        for k in range(run_len):
            hit &= dark[k : k + hit.size]
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return None
    return step_start + first_true + int(idx[0])


def _classify_hsv_1d(pix: np.ndarray, bright: bool) -> np.ndarray:
    """Classify a 1D slice of HSV pixels (N, 3) as walkable floor (or bright room)."""
    s = pix[:, 1]
    v = pix[:, 2]
    is_floor = (v >= 118) & (s <= 62)
    if bright:
        h = pix[:, 0]
        is_floor |= (v >= 135) & ((h < 90) | (h > 140))
    return is_floor


def _wall_scan_in_local_roi(
    hsv: np.ndarray, cx: float, cy: float, door_h: float, *, bright: bool
) -> tuple[float, float, float]:
    """Vectorized 1D HSV ray & cardinal wall scan (zero 2D mask allocation)."""
    h, w = hsv.shape[:2]
    max_r = int(min(w, h) * 0.30)
    min_r = max(10, int(door_h * 0.35))
    if min_r >= max_r:
        return cx, cy, max(16.0, door_h * 3.0)

    steps = np.arange(min_r, max_r, dtype=np.float64)
    diag: list[float] = []

    for deg in (200, 215, 225, 235, 250, 290, 305, 315, 325, 340):
        rad = math.radians(deg)
        dx, dy = math.cos(rad), math.sin(rad)
        px = np.rint(cx + dx * steps).astype(np.intp)
        py = np.rint(cy + dy * steps).astype(np.intp)
        oob = np.flatnonzero((px < 0) | (px >= w) | (py < 0) | (py >= h))
        if oob.size:
            cut = int(oob[0])
            if cut == 0:
                continue
            px = px[:cut]
            py = py[:cut]
        is_floor = _classify_hsv_1d(hsv[py, px], bright)
        hit_step = _first_wall_hit(is_floor, min_r, 5)
        if hit_step is not None:
            wx = cx + dx * hit_step
            wy = cy + dy * hit_step
            diag.append(max(abs(wx - cx), abs(wy - cy)))

    if not diag:
        return cx, cy, max(16.0, door_h * 3.0)

    rough_r = float(np.median(diag))

    def scan_wall(axis: str, sign: int, fracs: tuple[float, ...]) -> float:
        hits: list[float] = []
        step_start = max(2, int(0.25 * rough_r))
        step_end = max(step_start + 4, int(1.8 * rough_r))
        base = cx if axis == "x" else cy
        base_i = int(round(base))
        limit = w if axis == "x" else h
        other_base = cy if axis == "x" else cx
        other_limit = h if axis == "x" else w

        for f in fracs:
            other_i = int(round(other_base + f * rough_r))
            if other_i < 0 or other_i >= other_limit:
                continue
            if sign > 0:
                lo = base_i + step_start
                hi = min(limit, base_i + step_end)
                if lo < 0 or lo >= hi:
                    continue
                pix = hsv[other_i, lo:hi] if axis == "x" else hsv[lo:hi, other_i]
            else:
                hi = base_i - step_start
                lo = max(0, base_i - step_end + 1)
                if hi < 0 or hi >= limit or lo > hi:
                    continue
                pix = (
                    hsv[other_i, lo : hi + 1][::-1]
                    if axis == "x"
                    else hsv[lo : hi + 1, other_i][::-1]
                )
            is_floor = _classify_hsv_1d(pix, bright)
            hit_step = _first_wall_hit(is_floor, step_start, 4)
            if hit_step is not None:
                hits.append(base + sign * hit_step)
        return float(np.median(hits)) if hits else base + sign * rough_r

    x_l = scan_wall("x", -1, (-0.55, -0.45, -0.35, -0.25))
    x_r = scan_wall("x", +1, (-0.55, -0.45, -0.35, -0.25))
    y_t = scan_wall("y", -1, (-0.55, -0.45, -0.35, 0.35, 0.45, 0.55))

    room_size = max(16.0, x_r - x_l)
    rx = (x_l + x_r) / 2.0
    ry = y_t + room_size / 2.0
    return rx, ry, room_size


def measure_starting_room(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    door_h: float,
    *,
    use_wall_scan: bool = False,
    hsv: np.ndarray | None = None,
) -> tuple[float, float, float, tuple[float, float, float, float]]:
    """
    Measure the square starting room's center (rx, ry), width (room_size),
    and 4-way hallway exits (N, E, S, W) around the orange door centroid (cx, cy).
    Uses pillar-proof Chebyshev distance transform by default, or upper-half wall
    scanning when use_wall_scan=True (e.g., for aniimotools.dev reference images).
    """
    h, w = bgr.shape[:2]
    if hsv is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if use_wall_scan:
        rx, ry, room_size = _wall_scan_in_local_roi(hsv, cx, cy, door_h, bright=False)
        exit_fn = _room_floor_mask
    else:
        win = max(90, int(min(w, h) * 0.35))
        x0, x1 = max(0, int(round(cx)) - win), min(w, int(round(cx)) + win)
        y0, y1 = max(0, int(round(cy)) - win), min(h, int(round(cy)) + win)
        sub_bright = _room_bright_mask(hsv[y0:y1, x0:x1])
        sh, sw = sub_bright.shape

        ff = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
        ff[1:-1, 1:-1] = sub_bright
        ff_mask = np.zeros((sh + 4, sw + 4), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 2)
        solid_padded = (ff != 2).view(np.uint8)
        solid = solid_padded[1:-1, 1:-1]

        dist_c = cv2.distanceTransform(solid_padded, cv2.DIST_C, 3)[1:-1, 1:-1]
        lcx, lcy = cx - x0, cy - y0
        srad = max(35.0, min(door_h * 2.2, min(w, h) * 0.08))
        irad = int(math.ceil(srad))
        bx0 = max(0, int(math.floor(lcx)) - irad)
        bx1 = min(sw, int(math.ceil(lcx)) + irad + 1)
        by0 = max(0, int(math.floor(lcy)) - irad)
        by1 = min(sh, int(math.ceil(lcy)) + irad + 1)

        peak_val = 0.0
        candidates: list[tuple[float, float, float]] = []
        if bx1 > bx0 and by1 > by0:
            sub_dist = dist_c[by0:by1, bx0:bx1]
            yy, xx = np.ogrid[by0:by1, bx0:bx1]
            near_mask = ((xx - lcx) ** 2 + (yy - lcy) ** 2) <= (srad * srad)
            masked_dist = np.where(near_mask, sub_dist, 0.0)
            peak_val = float(masked_dist.max())

            if 12.0 <= peak_val <= min(sh, sw):
                py_idx, px_idx = np.nonzero(masked_dist >= peak_val - 0.5)
                rx_loc = float(np.mean(px_idx)) + bx0
                ry_loc = float(np.mean(py_idx)) + by0
                snapped_x, snapped_y, snapped_sz = _snap_flat_walls_from_solid(
                    solid, rx_loc, ry_loc, peak_val
                )
                candidates.append((x0 + snapped_x, y0 + snapped_y, snapped_sz))

        if candidates:
            rx, ry, room_size = candidates[0]
            valid_c, corners_c = _validate_room_corners_from_hsv(hsv, rx, ry, room_size / 2.0)
            if not valid_c:
                ws_rx, ws_ry, ws_sz = _wall_scan_in_local_roi(hsv, cx, cy, door_h, bright=True)
                _, corners_ws = _validate_room_corners_from_hsv(hsv, ws_rx, ws_ry, ws_sz / 2.0)
                if sum(corners_ws) < sum(corners_c):
                    rx, ry, room_size = ws_rx, ws_ry, ws_sz
        else:
            rx, ry, room_size = _wall_scan_in_local_roi(hsv, cx, cy, door_h, bright=True)
        exit_fn = _room_bright_mask

    half = room_size / 2.0
    exits: list[float] = []
    for ex0, ey0, ex1, ey1 in _exit_probe_boxes(rx, ry, half, w, h):
        if ex1 <= ex0 or ey1 <= ey0:
            exits.append(0.0)
        else:
            probe_mask = exit_fn(hsv[ey0:ey1, ex0:ex1])
            exits.append(cv2.countNonZero(probe_mask) / float((ex1 - ex0) * (ey1 - ey0)))

    return rx, ry, room_size, (exits[0], exits[1], exits[2], exits[3])


def _downscale_if_needed(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale image so its longest side is at most max_dim using area-preserving resampling."""
    if max_dim <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # For >2x downscales (e.g. 4K -> 1440), a fast 2x integer box-filter step first
    # avoids OpenCV's slow generic non-integer INTER_AREA kernel (~2.2x faster).
    if scale <= 0.5 and w >= 4 and h >= 4:
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _detect_doors_in_roi(
    roi: np.ndarray,
    hsv: np.ndarray,
    floor_mask: np.ndarray,
    ranges: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]],
    return_debug: bool,
) -> tuple[DoorSignature | None, np.ndarray | None, dict[str, Any]]:
    rh, rw = roi.shape[:2]
    total_px = float(max(1, rw * rh))

    min_area = max(20.0, total_px * 0.000010)
    max_area = total_px * 0.08
    min_sep = max(20.0, min(rw, rh) * 0.02)

    blue_lo, blue_hi = ranges["blue"]
    raw_blue = cv2.inRange(hsv, np.asarray(blue_lo, dtype=np.uint8), np.asarray(blue_hi, dtype=np.uint8))
    blue_mask = None
    blue_hit = None
    if cv2.countNonZero(raw_blue) >= 10:
        blue_mask = _clean_mask(raw_blue)
        blue_hit = _find_best_door_contour(
            blue_mask, floor_mask, rw, rh, min_area=min_area, max_area=max_area
        )

    orange_lo, orange_hi = ranges["orange"]
    raw_orange = cv2.inRange(
        hsv, np.asarray(orange_lo, dtype=np.uint8), np.asarray(orange_hi, dtype=np.uint8)
    )
    orange_mask = None
    orange_hit = None
    if cv2.countNonZero(raw_orange) >= 10:
        orange_mask = _clean_mask(raw_orange)
        exclude_pt = blue_hit[1] if blue_hit is not None else None
        orange_hit = _find_best_door_contour(
            orange_mask,
            floor_mask,
            rw,
            rh,
            min_area=min_area,
            max_area=max_area,
            exclude_center=exclude_pt,
            min_sep_px=min_sep,
        )

    used_yellow_for = None
    yel_mask = None
    if (orange_hit is None) != (blue_hit is None) and "yellow_fallback" in ranges:
        yel_lo, yel_hi = ranges["yellow_fallback"]
        raw_yel = cv2.inRange(
            hsv, np.asarray(yel_lo, dtype=np.uint8), np.asarray(yel_hi, dtype=np.uint8)
        )
        if cv2.countNonZero(raw_yel) >= 10:
            yel_mask = _clean_mask(raw_yel)
            if orange_hit is None:
                orange_hit = _find_best_door_contour(
                    yel_mask,
                    floor_mask,
                    rw,
                    rh,
                    min_area=min_area,
                    max_area=max_area,
                    exclude_center=blue_hit[1] if blue_hit is not None else None,
                    min_sep_px=min_sep,
                    min_solidity=0.22,
                )
                if orange_hit is not None:
                    used_yellow_for = "orange"
            elif blue_hit is None:
                blue_hit = _find_best_door_contour(
                    yel_mask,
                    floor_mask,
                    rw,
                    rh,
                    min_area=min_area,
                    max_area=max_area,
                    exclude_center=orange_hit[1],
                    min_sep_px=min_sep,
                    min_solidity=0.22,
                )
                if blue_hit is not None:
                    used_yellow_for = "blue"

    info: dict[str, Any] = {
        "roi_size": (rw, rh),
        "blue_found": blue_hit is not None,
        "orange_found": orange_hit is not None,
        "used_yellow_fallback": used_yellow_for,
    }

    sig: DoorSignature | None = None
    exit_room_box: tuple[float, float, float] | None = None
    start_corners: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    if blue_hit is not None and orange_hit is not None:
        _, (ox, oy), o_area, (ow, oh) = orange_hit
        _, (bx, by), b_area, (bw, bh) = blue_hit

        rx, ry, room_size, exits = measure_starting_room(roi, ox, oy, float(oh), hsv=hsv)
        start_corners_valid, start_corners = _validate_room_corners(floor_mask, rx, ry, room_size / 2.0)
        orange_floor_surround = _has_floor_surround(floor_mask, ox, oy, radius=max(ow, oh) * 1.5)
        start_valid = (
            start_corners_valid
            and room_size >= max(ow, oh) * 2.5
            and orange_floor_surround >= 0.35
        )

        started_from = "orange"
        target_x, target_y = bx, by

        blue_floor_surround = _has_floor_surround(floor_mask, bx, by, radius=max(bw, bh) * 1.5)
        if blue_floor_surround >= 0.35:
            ex_rx, ex_ry, ex_room_size, ex_exits = measure_starting_room(roi, bx, by, float(bh), hsv=hsv)
            exit_corners_valid, ex_corners = _validate_room_corners(floor_mask, ex_rx, ex_ry, ex_room_size / 2.0)
            exit_valid = exit_corners_valid and ex_room_size >= max(bw, bh) * 2.5
            exit_room_box = (ex_rx, ex_ry, ex_room_size)
            if not start_valid and exit_valid:
                started_from = "blue"
                rx, ry, room_size, exits = ex_rx, ex_ry, ex_room_size, ex_exits
                start_valid, start_corners = exit_valid, ex_corners
                target_x, target_y = ox, oy
            elif start_valid and exit_valid and abs(ex_room_size - room_size) / max(room_size, 1e-6) <= 0.15:
                room_size = (room_size + ex_room_size) / 2.0
            info["exit_room_size_px"] = round(ex_room_size, 1)
            info["exit_room_corner_valid"] = exit_valid

        pixel_dist = math.hypot(target_x - rx, target_y - ry)
        norm_dist = pixel_dist / max(room_size, 1e-6)
        angle_deg = math.degrees(math.atan2(target_y - ry, target_x - rx))
        size_ratio = o_area / max(b_area, 1e-6)

        sig = DoorSignature(
            normalized_distance=norm_dist,
            angle_deg=angle_deg,
            size_ratio=size_ratio,
            orange_centroid=(ox, oy),
            blue_centroid=(bx, by),
            orange_area=o_area,
            blue_area=b_area,
            orange_size_px=(ow, oh),
            blue_size_px=(bw, bh),
            room_center=(rx, ry),
            room_size=room_size,
            exits=exits,
            started_from=started_from,
        )
        info["signature"] = sig.to_dict()
        info["started_from"] = started_from
        info["orange_size_px"] = (ow, oh)
        info["blue_size_px"] = (bw, bh)
        info["room_size_px"] = round(room_size, 1)
        info["room_corner_valid"] = start_valid
        info["room_outer_corners"] = [round(v, 2) for v in start_corners]
        info["exits"] = [round(v, 2) for v in exits]
        if return_debug:
            o_mask = yel_mask if used_yellow_for == "orange" and yel_mask is not None else orange_mask
            b_mask = yel_mask if used_yellow_for == "blue" and yel_mask is not None else blue_mask
            if o_mask is not None:
                info["orange_hsv_mean"] = tuple(round(float(v), 1) for v in cv2.mean(hsv, mask=o_mask)[:3])
            if b_mask is not None:
                info["blue_hsv_mean"] = tuple(round(float(v), 1) for v in cv2.mean(hsv, mask=b_mask)[:3])

    overlay: np.ndarray | None = None
    if return_debug:
        overlay = roi.copy()
        if blue_hit is not None:
            b_cnt, (bx, by), b_area, (bw, bh) = blue_hit
            cv2.drawContours(overlay, [b_cnt], -1, (255, 200, 0), 2)
            cv2.circle(overlay, (int(round(bx)), int(round(by))), 5, (255, 255, 0), -1)
            cv2.putText(
                overlay,
                f"BLUE {bw}x{bh}px ({int(b_area)})",
                (int(round(bx)) + 8, int(round(by)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2,
            )
        if orange_hit is not None:
            o_cnt, (ox, oy), o_area, (ow, oh) = orange_hit
            cv2.drawContours(overlay, [o_cnt], -1, (0, 140, 255), 2)
            cv2.circle(overlay, (int(round(ox)), int(round(oy))), 5, (0, 180, 255), -1)
            cv2.putText(
                overlay,
                f"ORANGE {ow}x{oh}px ({int(o_area)})",
                (int(round(ox)) + 8, int(round(oy)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
            )
        if sig is not None:
            rx, ry = sig.room_center
            half = sig.room_size / 2.0
            cv2.rectangle(
                overlay,
                (int(round(rx - half)), int(round(ry - half))),
                (int(round(rx + half)), int(round(ry + half))),
                (255, 255, 255),
                2,
            )
            cv2.circle(overlay, (int(round(rx)), int(round(ry))), 4, (255, 255, 255), -1)

            for (x0, y0, x1, y1), ex_val in zip(
                _exit_probe_boxes(rx, ry, half, rw, rh), sig.exits
            ):
                color = (0, 230, 80) if ex_val >= 0.5 else (40, 40, 230)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

            for (cx0, cy0, cx1, cy1), c_val in zip(
                _corner_probe_boxes(rx, ry, half, rw, rh), start_corners
            ):
                c_color = (255, 220, 0) if c_val < 0.20 else (255, 0, 255)
                cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), c_color, 2)

            if exit_room_box is not None and sig.started_from != "blue":
                erx, ery, ersz = exit_room_box
                ehalf = ersz / 2.0
                cv2.rectangle(
                    overlay,
                    (int(round(erx - ehalf)), int(round(ery - ehalf))),
                    (int(round(erx + ehalf)), int(round(ery + ehalf))),
                    (255, 190, 90),
                    2,
                )

            p1 = (int(round(rx)), int(round(ry)))
            target_pt = sig.orange_centroid if sig.started_from == "blue" else sig.blue_centroid
            p2 = (int(round(target_pt[0])), int(round(target_pt[1])))
            cv2.line(overlay, p1, p2, (0, 255, 128), 2)
            exits_txt = ",".join(f"{v:.2f}" for v in sig.exits)
            label = (
                f"dist={sig.normalized_distance:.2f}  "
                f"ang={sig.angle_deg:.1f}deg  "
                f"room={sig.room_size:.0f}px  "
                f"exits=[{exits_txt}]"
            )
            cv2.putText(
                overlay,
                label,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 128),
                2,
            )

    return sig, overlay, info


def detect_doors(
    image_bgr: np.ndarray,
    trim_margin: float = 0.05,
    hsv_ranges: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
    return_debug: bool = False,
    max_dim: int = 1440,
    retry_untrimmed: bool = False,
) -> tuple[DoorSignature | None, np.ndarray | None, dict[str, Any]]:
    """
    Detect the orange and blue Sanctum doors and return:
      (DoorSignature | None, debug_overlay_bgr | None, info_dict)
    When retry_untrimmed=True and trim_margin > 0, downscales once and retries
    without trimming if the trimmed pass fails, reusing the downscaled frame/HSV/floor.
    """
    ranges = hsv_ranges or DEFAULT_HSV_RANGES

    if retry_untrimmed and trim_margin > 0.0:
        full_roi = _downscale_if_needed(image_bgr, max_dim)
        full_hsv = cv2.cvtColor(full_roi, cv2.COLOR_BGR2HSV)
        full_floor = _floor_mask_from_hsv(full_hsv)

        fh, fw = full_roi.shape[:2]
        x0 = int(round(fw * trim_margin))
        x1 = int(round(fw * (1.0 - trim_margin)))
        y0 = int(round(fh * trim_margin))
        y1 = int(round(fh * (1.0 - trim_margin)))
        if x1 > x0 and y1 > y0:
            sig, overlay, info = _detect_doors_in_roi(
                full_roi[y0:y1, x0:x1],
                full_hsv[y0:y1, x0:x1],
                full_floor[y0:y1, x0:x1],
                ranges,
                return_debug,
            )
            if sig is not None:
                return sig, overlay, info
            retry_sig, retry_overlay, retry_info = _detect_doors_in_roi(
                full_roi, full_hsv, full_floor, ranges, return_debug
            )
            if retry_sig is not None:
                return retry_sig, retry_overlay, retry_info
            retry_doors = int(retry_info["orange_found"]) + int(retry_info["blue_found"])
            trim_doors = int(info["orange_found"]) + int(info["blue_found"])
            if retry_doors > trim_doors:
                return retry_sig, retry_overlay, retry_info
            return sig, overlay, info

        return _detect_doors_in_roi(full_roi, full_hsv, full_floor, ranges, return_debug)

    roi = crop_safe_margin(image_bgr, margin=trim_margin)
    roi = _downscale_if_needed(roi, max_dim)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    floor_mask = _floor_mask_from_hsv(hsv)
    return _detect_doors_in_roi(roi, hsv, floor_mask, ranges, return_debug)
