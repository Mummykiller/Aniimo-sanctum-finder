"""
Remote layouts.json synchronization into local .cache/ with offline fallback.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from urls import REMOTE_LAYOUTS_URL

BASE_DIR = Path(__file__).resolve().parent
BUNDLED_LAYOUTS_PATH = BASE_DIR / "layouts.json"
CACHE_DIR = BASE_DIR / ".cache"
CACHED_LAYOUTS_PATH = CACHE_DIR / "layouts.json"
MAX_REMOTE_BYTES = 1_048_576  # 1 MiB safety cap


def _is_valid_layouts_db(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    layouts = data.get("layouts")
    if not isinstance(layouts, dict) or not layouts:
        return False
    for key, ref in layouts.items():
        if not isinstance(ref, dict):
            return False
        try:
            if int(key) <= 0:
                return False
            ang = float(ref["angle_deg"])
            dist = float(ref["normalized_distance"])
        except (KeyError, TypeError, ValueError):
            return False
        if not (math.isfinite(ang) and math.isfinite(dist) and dist > 0.0):
            return False
    return True


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if _is_valid_layouts_db(data):
            return data
    except Exception:
        pass
    return None


def _version_of(data: dict[str, Any] | None) -> int:
    if not data:
        return -1
    raw = data.get("version", 0)
    if isinstance(raw, bool):
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return 0


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2) + "\n"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(payload)
            fh.flush()
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def sync_layouts(remote_url: str = REMOTE_LAYOUTS_URL, timeout: float = 2.0) -> dict[str, Any]:
    """
    Fetch remote layouts.json if newer than local cache/bundled copy.
    Always falls back cleanly to .cache/layouts.json or bundled layouts.json offline.
    """
    bundled = _read_json(BUNDLED_LAYOUTS_PATH)
    cached = _read_json(CACHED_LAYOUTS_PATH)

    active = cached if _version_of(cached) >= _version_of(bundled) else bundled
    if active is None:
        active = {"version": 0, "layouts": {}}

    if not remote_url or "YOUR_USERNAME" in remote_url:
        return active

    try:
        req = urllib.request.Request(remote_url, headers={"User-Agent": "aniimo-sanctum-finder/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read(MAX_REMOTE_BYTES + 1)
        if len(raw_bytes) > MAX_REMOTE_BYTES:
            return active
        remote_data = json.loads(raw_bytes.decode("utf-8"))
        if _is_valid_layouts_db(remote_data) and _version_of(remote_data) > _version_of(active):
            try:
                _atomic_write_json(CACHED_LAYOUTS_PATH, remote_data)
            except OSError:
                pass
            return remote_data
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ValueError, OSError):
        pass

    return active
