"""
URL & Remote Sync Configuration for Aniimo Sanctum Launcher.
"""

from __future__ import annotations

# Replace with your public GitHub raw URL once published:
REMOTE_LAYOUTS_URL = (
    "https://raw.githubusercontent.com/Mummykiller/Aniimo-sanctum-finder/main/layouts.json"
)

BASE_SANCTUM_URL = "https://aniimotools.dev/map/lost-sanctum/"

KNOWN_LAYOUT_IDS: tuple[int, ...] = (
    31, 32, 34, 35, 36, 37, 39, 40,
    41, 42, 45, 48, 49, 50,
    53, 54, 55, 56, 57, 58, 59,
    61, 62,
)

LAYOUT_URLS: dict[int, str] = {
    layout_id: f"{BASE_SANCTUM_URL}#plan-{layout_id}" for layout_id in KNOWN_LAYOUT_IDS
}


def get_url_for_layout(layout_id: int) -> str:
    """Return the destination URL for a matched integer layout ID."""
    lid = int(layout_id)
    if lid <= 0:
        return ""
    return LAYOUT_URLS.get(lid, f"{BASE_SANCTUM_URL}#plan-{lid}")
