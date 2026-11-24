#!/usr/bin/env python3
"""
Developer Helper: Build layouts.json from numbered reference images
in Images/<difficulty>/*.png (or local_reference_images/*.png).

Uses multi-scale template matching on clean Main/Side entrance door icons
paired with starting-room scale-ruler measurement so small keys/eggs/chests
never cause false door detections. Never commits or distributes images.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from vision import measure_starting_room

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "Images"
OUTPUT_JSON_PATH = BASE_DIR / "layouts.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):08d}")
    except ValueError:
        return (1, path.stem)


def _crop_door_from_region(
    bgr: np.ndarray, mask: np.ndarray, rx0: int, rx1: int, ry0: int, ry1: int
) -> np.ndarray | None:
    sub = mask[ry0:ry1, rx0:rx1]
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    cx = rx0 + x + bw // 2
    cy = ry0 + y + bh // 2
    pw = max(12, int(bw * 1.6))
    ph = max(14, int(bh * 1.6))
    h, w = bgr.shape[:2]
    x0, x1 = max(0, cx - pw // 2), min(w, cx + pw // 2)
    y0, y1 = max(0, cy - ph // 2), min(h, cy + ph // 2)
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1].copy()


def _extract_pill_door_templates(seed_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect Main & Side entrance templates via white icon border + dark pill below."""
    h, w = seed_bgr.shape[:2]
    hsv = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2HSV)
    floor = ((hsv[:, :, 2] >= 120) & (hsv[:, :, 1] <= 65)).astype(np.uint8)
    dark = (hsv[:, :, 2] < 65).astype(np.uint8)
    white = ((hsv[:, :, 2] >= 220) & (hsv[:, :, 1] <= 45)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def best_crop(mask: np.ndarray) -> np.ndarray | None:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1.0
        best_box: tuple[int, int, int, int] | None = None
        for cnt in cnts:
            area = float(cv2.contourArea(cnt))
            if area < 30:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 6 or bh < 8 or bw > w * 0.08 or bh > h * 0.10:
                continue
            asp = bw / float(bh)
            if asp < 0.45 or asp > 1.35:
                continue
            mm = cv2.moments(cnt)
            if mm["m00"] <= 1e-6:
                continue
            cx, cy = mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]
            rx0, rx1 = max(0, x - 4), min(w, x + bw + 4)
            ry0, ry1 = max(0, y - 5), min(h, y + bh + 4)
            white_ratio = float(np.mean(white[ry0:ry1, rx0:rx1]))
            px0, px1 = max(0, int(cx - bw * 1.4)), min(w, int(cx + bw * 1.4))
            py0, py1 = min(h - 1, int(y + bh * 1.15)), min(h, int(y + bh * 2.35))
            pill_dark = float(np.mean(dark[py0:py1, px0:px1])) if py1 > py0 and px1 > px0 else 0.0
            fx0, fx1 = max(0, int(cx - bh * 1.8)), min(w, int(cx + bh * 1.8))
            fy0, fy1 = max(0, int(cy - bh * 1.8)), min(h, int(cy + bh * 1.8))
            fl_ratio = float(np.mean(floor[fy0:fy1, fx0:fx1]))
            if pill_dark < 0.25 or white_ratio < 0.06:
                continue
            score = area * (0.2 + white_ratio) * (0.2 + pill_dark) * (0.2 + fl_ratio)
            if score > best_score:
                best_score = score
                best_box = (int(cx), int(cy), bw, bh)
        if best_box is None:
            return None
        cx, cy, bw, bh = best_box
        pw, ph = max(12, int(bw * 1.6)), max(14, int(bh * 1.6))
        x0, x1 = max(0, cx - pw // 2), min(w, cx + pw // 2)
        y0, y1 = max(0, cy - ph // 2), min(h, cy + ph // 2)
        return seed_bgr[y0:y1, x0:x1].copy()

    orange_mask = cv2.inRange(hsv, np.array((10, 60, 130)), np.array((26, 215, 255)))
    blue_mask = cv2.inRange(hsv, np.array((98, 70, 120)), np.array((128, 255, 255)))
    tpl_m = best_crop(orange_mask)
    tpl_s = best_crop(blue_mask)
    if tpl_m is None or tpl_s is None:
        return None
    return tpl_m, tpl_s


def _obtain_door_templates(input_dir: Path, files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    seed_56 = input_dir / "beginner" / "56.png"
    if seed_56.is_file():
        bgr = cv2.imread(str(seed_56))
        if bgr is not None:
            h, w = bgr.shape[:2]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            om = cv2.inRange(hsv, np.array((10, 60, 130)), np.array((26, 215, 255)))
            bm = cv2.inRange(hsv, np.array((98, 70, 120)), np.array((128, 255, 255)))
            tpl_m = _crop_door_from_region(bgr, om, 0, int(w * 0.20), int(h * 0.40), int(h * 0.65))
            tpl_s = _crop_door_from_region(bgr, bm, int(w * 0.25), int(w * 0.45), int(h * 0.40), int(h * 0.65))
            if tpl_m is not None and tpl_s is not None:
                return tpl_m, tpl_s

    for p in files:
        if not p.stem.isdigit():
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        pair = _extract_pill_door_templates(bgr)
        if pair is not None:
            return pair

    raise RuntimeError(f"Could not extract Main/Side entrance door templates from {input_dir}")


def _match_door_multiscale(
    bgr: np.ndarray, tpl: np.ndarray
) -> tuple[float, tuple[float, float], int]:
    best_val = -1.0
    best_pt = (0.0, 0.0)
    best_h = 0
    th, tw = tpl.shape[:2]

    for sc in np.linspace(0.55, 1.35, 17):
        nw = max(8, int(round(tw * float(sc))))
        nh = max(8, int(round(th * float(sc))))
        if nw >= bgr.shape[1] or nh >= bgr.shape[0]:
            continue
        rt = cv2.resize(tpl, (nw, nh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(bgr, rt, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = float(max_val)
            best_pt = (max_loc[0] + nw / 2.0, max_loc[1] + nh / 2.0)
            best_h = nh

    return best_val, best_pt, best_h


def build_layouts_db(input_dir: Path, output_path: Path = OUTPUT_JSON_PATH) -> dict:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Reference directory not found: {input_dir}\n"
            "Place numbered images (31.png, 53.png, ...) inside Images/<tier>/ or pass --input-dir."
        )

    prev_version = 0
    if output_path.exists():
        try:
            prev_version = int(json.loads(output_path.read_text(encoding="utf-8")).get("version", 0))
        except Exception:
            prev_version = 0

    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    files.sort(key=_sort_key)

    tpl_main, tpl_side = _obtain_door_templates(input_dir, files)

    layouts: dict[str, dict] = {}
    for img_path in files:
        if not img_path.stem.isdigit():
            print(f"[skip] Non-numeric filename: {img_path.relative_to(input_dir)}")
            continue

        if img_path.stem in layouts:
            # Shared layout ID across difficulty folders (e.g., nightmare & chaos)
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[warn] Could not read image: {img_path}")
            continue

        vm, pm, hm = _match_door_multiscale(img_bgr, tpl_main)
        vs, ps, _ = _match_door_multiscale(img_bgr, tpl_side)
        if vm < 0.75 or vs < 0.75:
            print(
                f"[warn] Failed to detect both doors in {img_path.relative_to(input_dir)} "
                f"(main_corr={vm:.2f}, side_corr={vs:.2f})"
            )
            continue

        rx, ry, room_size, exits = measure_starting_room(
            img_bgr, pm[0], pm[1], float(hm), use_wall_scan=True
        )
        norm_dist = math.hypot(ps[0] - rx, ps[1] - ry) / max(room_size, 1e-6)
        angle_deg = math.degrees(math.atan2(ps[1] - ry, ps[0] - rx))

        tier = img_path.parent.name if img_path.parent != input_dir else "default"
        entry = {
            "normalized_distance": round(norm_dist, 4),
            "angle_deg": round(angle_deg, 3),
            "size_ratio": 1.0,
            "exits": [round(v, 3) for v in exits],
            "tier": tier,
        }
        layouts[img_path.stem] = entry
        print(
            f"[ok] Layout {img_path.stem} ({tier}): "
            f"tm=({vm:.2f},{vs:.2f}) rsize={room_size:.0f}px "
            f"dist={entry['normalized_distance']}, ang={entry['angle_deg']}°, "
            f"exits={entry['exits']}"
        )

    db = {
        "version": prev_version + 1,
        "layouts": layouts,
    }
    output_path.write_text(json.dumps(db, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(layouts)} layout signatures (version {db['version']}) -> {output_path}")
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="Build layouts.json from local reference images")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing numbered reference images (default: Images/)",
    )
    args = parser.parse_args()
    build_layouts_db(args.input_dir)


if __name__ == "__main__":
    main()
