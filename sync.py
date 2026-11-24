"""
Remote layouts.json synchronization into local .cache/ with offline fallback.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from urls import REMOTE_LAYOUTS_URL

BASE_DIR = Path(__file__).resolve().parent
BUNDLED_LAYOUTS_PATH = BASE_DIR / "layouts.json"
CACHE_DIR = BASE_DIR / ".cache"
CACHED_LAYOUTS_PATH = CACHE_DIR / "layouts.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("layouts"), dict):
            return data
    except Exception:
        pass
    return None


def _version_of(data: dict[str, Any] | None) -> int:
    if not data:
        return -1
    try:
        return int(data.get("version", 0))
    except (TypeError, ValueError):
        return 0


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
            remote_data = json.loads(resp.read().decode("utf-8"))
        if isinstance(remote_data, dict) and isinstance(remote_data.get("layouts"), dict):
            if _version_of(remote_data) > _version_of(active):
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                CACHED_LAYOUTS_PATH.write_text(
                    json.dumps(remote_data, indent=2) + "\n", encoding="utf-8"
                )
                return remote_data
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass

    return active
