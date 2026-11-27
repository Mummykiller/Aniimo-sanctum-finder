"""Unit tests for sync.py and notify.py."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import notify
import sync


def _valid_db(version: int) -> dict:
    return {
        "version": version,
        "layouts": {
            "40": {"angle_deg": 45.0, "normalized_distance": 2.5, "exits": [1.0, 0.0, 0.0, 0.0]},
        },
    }


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        return self._payload if n < 0 else self._payload[:n]


class TestSync(unittest.TestCase):
    def test_version_of_edge_cases(self) -> None:
        self.assertEqual(sync._version_of(None), -1)
        self.assertEqual(sync._version_of({"version": 9}), 9)
        self.assertEqual(sync._version_of({"version": "12"}), 12)
        self.assertEqual(sync._version_of({"version": True}), 0)
        self.assertEqual(sync._version_of({"version": float("inf")}), 0)
        self.assertEqual(sync._version_of({"version": float("nan")}), 0)
        self.assertEqual(sync._version_of({"version": "bad"}), 0)

    def test_is_valid_layouts_db(self) -> None:
        self.assertTrue(sync._is_valid_layouts_db(_valid_db(1)))
        self.assertFalse(sync._is_valid_layouts_db({"version": 1, "layouts": {}}))
        self.assertFalse(sync._is_valid_layouts_db({"version": 1, "layouts": {"0": {"angle_deg": 0, "normalized_distance": 1}}}))
        self.assertFalse(sync._is_valid_layouts_db({"version": 1, "layouts": {"-1": {"angle_deg": 0, "normalized_distance": 1}}}))
        self.assertFalse(sync._is_valid_layouts_db({"version": 1, "layouts": {"1": {"angle_deg": float("nan"), "normalized_distance": 1}}}))
        self.assertFalse(sync._is_valid_layouts_db({"version": 1, "layouts": {"1": {"angle_deg": 0, "normalized_distance": 0}}}))

    def test_sync_updates_cache_atomically_and_rejects_poisoned_remote(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bundled_path = tmp / "layouts.json"
            cache_dir = tmp / ".cache"
            cache_path = cache_dir / "layouts.json"
            bundled_path.write_text(json.dumps(_valid_db(5)), encoding="utf-8")

            with mock.patch.object(sync, "BUNDLED_LAYOUTS_PATH", bundled_path), \
                 mock.patch.object(sync, "CACHE_DIR", cache_dir), \
                 mock.patch.object(sync, "CACHED_LAYOUTS_PATH", cache_path):
                # 1. Newer valid remote updates cache
                v6 = json.dumps(_valid_db(6)).encode("utf-8")
                with mock.patch("sync.urllib.request.urlopen", return_value=_FakeResponse(v6)):
                    res = sync.sync_layouts("https://example.invalid/layouts.json")
                self.assertEqual(res["version"], 6)
                self.assertTrue(cache_path.exists())
                self.assertEqual(json.loads(cache_path.read_text(encoding="utf-8"))["version"], 6)

                # 2. Poisoned empty remote with higher version is rejected; cache stays v6
                poison = json.dumps({"version": 99, "layouts": {}}).encode("utf-8")
                with mock.patch("sync.urllib.request.urlopen", return_value=_FakeResponse(poison)):
                    res2 = sync.sync_layouts("https://example.invalid/layouts.json")
                self.assertEqual(res2["version"], 6)

                # 3. Oversized payload (>1 MiB) is rejected
                huge = b" " * (sync.MAX_REMOTE_BYTES + 10)
                with mock.patch("sync.urllib.request.urlopen", return_value=_FakeResponse(huge)):
                    res3 = sync.sync_layouts("https://example.invalid/layouts.json")
                self.assertEqual(res3["version"], 6)

                # 4. IncompleteRead / URLError fall back cleanly to v6
                with mock.patch("sync.urllib.request.urlopen", side_effect=http.client.IncompleteRead(b"partial")):
                    res4 = sync.sync_layouts("https://example.invalid/layouts.json")
                self.assertEqual(res4["version"], 6)

                with mock.patch("sync.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
                    res5 = sync.sync_layouts("https://example.invalid/layouts.json")
                self.assertEqual(res5["version"], 6)


class TestNotify(unittest.TestCase):
    def test_headless_linux_returns_false_without_spawning(self) -> None:
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("notify.subprocess.run") as mock_run:
            self.assertFalse(notify.send_notification("Title", "Body"))
            mock_run.assert_not_called()

    def test_linux_notify_send_uses_separator_and_falls_back_to_kdialog(self) -> None:
        cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            cmds.append(cmd)
            rc = 1 if Path(cmd[0]).name == "notify-send" else 0
            return subprocess.CompletedProcess(cmd, rc)

        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True), \
             mock.patch("notify.shutil.which", side_effect=lambda n: f"/usr/bin/{n}" if n in ("notify-send", "kdialog") else None), \
             mock.patch("notify.subprocess.run", side_effect=fake_run):
            ok = notify.send_notification("-odd-title", "Hello")

        self.assertTrue(ok)
        self.assertEqual(len(cmds), 2)
        self.assertIn("--", cmds[0])
        self.assertIn("-t", cmds[0])
        self.assertEqual(Path(cmds[1][0]).name, "kdialog")

    def test_windows_passes_text_via_env_vars(self) -> None:
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch("notify.shutil.which", side_effect=lambda n: "C:/powershell.exe" if n == "powershell" else None), \
             mock.patch("notify.subprocess.run", side_effect=fake_run):
            ok = notify.send_notification("Title'\"$()", "Msg `test`")

        self.assertTrue(ok)
        self.assertEqual(captured["env"]["ANIIMO_NOTIFY_TITLE"], "Title'\"$()")
        self.assertEqual(captured["env"]["ANIIMO_NOTIFY_MSG"], "Msg `test`")

    def test_windows_atomic_write_retries_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "layouts.json"
            real_replace = os.replace
            attempts = {"n": 0}

            def flaky_replace(src, dst):
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise PermissionError("AV lock")
                return real_replace(src, dst)

            with mock.patch("sync.os.replace", side_effect=flaky_replace), \
                 mock.patch("sync.time.sleep"):
                sync._atomic_write_json(target, _valid_db(7))

            self.assertEqual(attempts["n"], 3)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["version"], 7)


if __name__ == "__main__":
    unittest.main()
