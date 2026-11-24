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
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)


def _floor_mask_from_hsv(hsv: np.ndarray) -> np.ndarray:
    """Isolate light-gray/tan walkable room & corridor floors from dark walls/fog."""
    return ((hsv[:, :, 2] >= 125) & (hsv[:, :, 1] <= 65)).astype(np.uint8)


def _has_floor_surround(floor_mask: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Return fraction of walkable floor pixels around (cx, cy) to reject off-map UI icons."""
    h, w = floor_mask.shape
    r = max(8, int(round(radius)))
    x0 = max(0, int(round(cx)) - r)
    x1 = min(w, int(round(cx)) + r + 1)
    y0 = max(0, int(round(cy)) - r)
    y1 = min(h, int(round(cy)) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.mean(floor_mask[y0:y1, x0:x1]))


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
            vals.append(float(np.mean(floor[y0:y1, x0:x1])))
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
        for f in fracs:
            off = f * half
            for step in range(step_start, step_end):
                if axis == "x":
                    px = int(round(rx_local + sign * step))
                    py = int(round(ry_local + off))
                else:
                    px = int(round(rx_local + off))
                    py = int(round(ry_local + sign * step))
                if px < 0 or px >= sw or py < 0 or py >= sh:
                    break
                if solid[py, px] == 0:
                    hits.append(base + sign * step)
                    break
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


def _measure_room_wall_scan(
    floor: np.ndarray, cx: float, cy: float, door_h: float
) -> tuple[float, float, float]:
    """Scan upper-half diagonal rays and cardinal walls against the inner floor edge."""
    h, w = floor.shape
    wall = (1 - floor).astype(np.uint8)

    max_r = int(min(w, h) * 0.30)
    min_r = max(10, int(door_h * 0.35))
    diag: list[float] = []

    for deg in (200, 215, 225, 235, 250, 290, 305, 315, 325, 340):
        rad = math.radians(deg)
        dx, dy = math.cos(rad), math.sin(rad)
        seen_floor = False
        dark_run = 0
        for step in range(min_r, max_r):
            px = int(round(cx + dx * step))
            py = int(round(cy + dy * step))
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            if floor[py, px]:
                seen_floor = True
                dark_run = 0
            elif wall[py, px] and seen_floor:
                dark_run += 1
                if dark_run >= 5:
                    wx = cx + dx * (step - 4)
                    wy = cy + dy * (step - 4)
                    diag.append(max(abs(wx - cx), abs(wy - cy)))
                    break
            else:
                dark_run = 0

    if not diag:
        return cx, cy, max(16.0, door_h * 3.0)

    rough_r = float(np.median(diag))

    def scan_wall(axis: str, sign: int, fracs: tuple[float, ...]) -> float:
        hits: list[float] = []
        step_start = max(2, int(0.25 * rough_r))
        step_end = max(step_start + 4, int(1.8 * rough_r))
        for f in fracs:
            off = f * rough_r
            seen_floor = False
            dark_run = 0
            for step in range(step_start, step_end):
                if axis == "x":
                    px = int(round(cx + sign * step))
                    py = int(round(cy + off))
                else:
                    px = int(round(cx + off))
                    py = int(round(cy + sign * step))
                if px < 0 or px >= w or py < 0 or py >= h:
                    break
                if floor[py, px]:
                    seen_floor = True
                    dark_run = 0
                elif wall[py, px] and seen_floor:
                    dark_run += 1
                    if dark_run >= 4:
                        base = cx if axis == "x" else cy
                        hits.append(base + sign * (step - 3))
                        break
                else:
                    dark_run = 0
        base = cx if axis == "x" else cy
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
) -> tuple[float, float, float, tuple[float, float, float, float]]:
    """
    Measure the square starting room's center (rx, ry), width (room_size),
    and 4-way hallway exits (N, E, S, W) around the orange door centroid (cx, cy).
    Uses pillar-proof Chebyshev distance transform by default, or upper-half wall
    scanning when use_wall_scan=True (e.g., for aniimotools.dev reference images).
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    floor = ((hsv[:, :, 2] >= 118) & (hsv[:, :, 1] <= 62)).astype(np.uint8)
    room_bright = (
        floor.astype(bool)
        | ((hsv[:, :, 2] >= 135) & ((hsv[:, :, 0] < 90) | (hsv[:, :, 0] > 140)))
    ).astype(np.uint8)

    if use_wall_scan:
        rx, ry, room_size = _measure_room_wall_scan(floor, cx, cy, door_h)
        exit_mask = floor
    else:
        win = max(90, int(min(w, h) * 0.35))
        x0, x1 = max(0, int(round(cx)) - win), min(w, int(round(cx)) + win)
        y0, y1 = max(0, int(round(cy)) - win), min(h, int(round(cy)) + win)
        sub = room_bright[y0:y1, x0:x1].copy()
        sh, sw = sub.shape

        padded = np.pad(sub, 1, mode="constant", constant_values=0)
        ff = padded.copy()
        ff_mask = np.zeros((sh + 4, sw + 4), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 2)
        solid = (ff[1:-1, 1:-1] != 2).astype(np.uint8)

        dist_c = cv2.distanceTransform(solid, cv2.DIST_C, 3)
        lcx, lcy = cx - x0, cy - y0
        yy, xx = np.ogrid[:sh, :sw]
        srad = max(35.0, min(door_h * 2.2, min(w, h) * 0.08))
        near_mask = ((xx - lcx) ** 2 + (yy - lcy) ** 2) <= (srad**2)
        masked_dist = np.where(near_mask, dist_c, 0.0)
        peak_val = float(masked_dist.max())

        candidates: list[tuple[float, float, float]] = []
        if peak_val >= 12.0:
            py_idx, px_idx = np.nonzero(masked_dist >= peak_val - 0.5)
            rx_loc = float(np.mean(px_idx))
            ry_loc = float(np.mean(py_idx))
            snapped_x, snapped_y, snapped_sz = _snap_flat_walls_from_solid(
                solid, rx_loc, ry_loc, peak_val
            )
            candidates.append((x0 + snapped_x, y0 + snapped_y, snapped_sz))

        if candidates:
            rx, ry, room_size = candidates[0]
            valid_c, corners_c = _validate_room_corners(floor, rx, ry, room_size / 2.0)
            if not valid_c:
                ws_rx, ws_ry, ws_sz = _measure_room_wall_scan(room_bright, cx, cy, door_h)
                _, corners_ws = _validate_room_corners(floor, ws_rx, ws_ry, ws_sz / 2.0)
                if sum(corners_ws) < sum(corners_c):
                    rx, ry, room_size = ws_rx, ws_ry, ws_sz
        else:
            rx, ry, room_size = _measure_room_wall_scan(room_bright, cx, cy, door_h)
        exit_mask = room_bright

    half = room_size / 2.0
    exits: list[float] = []
    for x0, y0, x1, y1 in _exit_probe_boxes(rx, ry, half, w, h):
        if x1 <= x0 or y1 <= y0:
            exits.append(0.0)
        else:
            exits.append(float(np.mean(exit_mask[y0:y1, x0:x1])))

    return rx, ry, room_size, (exits[0], exits[1], exits[2], exits[3])


def detect_doors(
    image_bgr: np.ndarray,
    trim_margin: float = 0.05,
    hsv_ranges: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
    return_debug: bool = False,
    max_dim: int = 1440,
) -> tuple[DoorSignature | None, np.ndarray | None, dict[str, Any]]:
    """
    Detect the orange and blue Sanctum doors and return:
      (DoorSignature | None, debug_overlay_bgr | None, info_dict)
    """
    ranges = hsv_ranges or DEFAULT_HSV_RANGES
    roi = crop_safe_margin(image_bgr, margin=trim_margin)
    rh, rw = roi.shape[:2]
    if max_dim > 0 and max(rw, rh) > max_dim:
        scale = max_dim / float(max(rw, rh))
        new_w = max(1, int(round(rw * scale)))
        new_h = max(1, int(round(rh * scale)))
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rh, rw = roi.shape[:2]
    total_px = float(max(1, rw * rh))

    min_area = max(20.0, total_px * 0.000010)
    max_area = total_px * 0.08

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    floor_mask = _floor_mask_from_hsv(hsv)
    min_sep = max(20.0, min(rw, rh) * 0.02)

    blue_lo, blue_hi = ranges["blue"]
    blue_mask = _clean_mask(cv2.inRange(hsv, np.array(blue_lo), np.array(blue_hi)))
    blue_hit = _find_best_door_contour(
        blue_mask, floor_mask, rw, rh, min_area=min_area, max_area=max_area
    )

    orange_lo, orange_hi = ranges["orange"]
    orange_mask = _clean_mask(cv2.inRange(hsv, np.array(orange_lo), np.array(orange_hi)))
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
    if (orange_hit is None or blue_hit is None) and "yellow_fallback" in ranges:
        yel_lo, yel_hi = ranges["yellow_fallback"]
        yel_mask = _clean_mask(cv2.inRange(hsv, np.array(yel_lo), np.array(yel_hi)))
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

        rx, ry, room_size, exits = measure_starting_room(roi, ox, oy, float(oh))
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
            ex_rx, ex_ry, ex_room_size, ex_exits = measure_starting_room(roi, bx, by, float(bh))
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
        info["orange_hsv_mean"] = tuple(
            round(float(v), 1)
            for v in cv2.mean(
                hsv,
                mask=yel_mask if used_yellow_for == "orange" and yel_mask is not None else orange_mask,
            )[:3]
        )
        info["blue_hsv_mean"] = tuple(
            round(float(v), 1)
            for v in cv2.mean(
                hsv,
                mask=yel_mask if used_yellow_for == "blue" and yel_mask is not None else blue_mask,
            )[:3]
        )

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
