"""
Cross-platform desktop notifications for Aniimo Sanctum Finder.

Uses built-in OS tools with zero extra Python dependencies:
  - Linux: notify-send (libnotify), with kdialog / zenity fallbacks
  - Windows: PowerShell WinRT Toast notification, with NotifyIcon balloon fallback
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

APP_NAME = "Aniimo Sanctum Finder"

_WIN_POWERSHELL_NOTIFY = r"""
$t = $env:ANIIMO_NOTIFY_TITLE
$m = $env:ANIIMO_NOTIFY_MSG
try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    $xt = [System.Security.SecurityElement]::Escape($t)
    $xm = [System.Security.SecurityElement]::Escape($m)
    $xml = "<toast><visual><binding template='ToastGeneric'><text>$xt</text><text>$xm</text></binding></visual></toast>"
    $doc = New-Object Windows.Data.Xml.Dom.XmlDocument
    $doc.LoadXml($xml)
    $toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $n = New-Object System.Windows.Forms.NotifyIcon
    $n.Icon = [System.Drawing.SystemIcons]::Warning
    $n.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Warning
    $n.BalloonTipTitle = $t
    $n.BalloonTipText = $m
    $n.Visible = $true
    $n.ShowBalloonTip(5000)
    Start-Sleep -Milliseconds 1500
    $n.Dispose()
}
""".strip()


def send_notification(title: str, message: str) -> bool:
    """Send a non-fatal desktop notification on Linux or Windows. Returns True if dispatched."""
    try:
        if sys.platform.startswith("win"):
            ps = shutil.which("powershell") or shutil.which("pwsh")
            if not ps:
                return False
            env = os.environ.copy()
            env["ANIIMO_NOTIFY_TITLE"] = title
            env["ANIIMO_NOTIFY_MSG"] = message
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            res = subprocess.run(
                [ps, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", _WIN_POWERSHELL_NOTIFY],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=6,
                creationflags=creationflags,
                check=False,
            )
            return res.returncode == 0

        if sys.platform.startswith("linux") or os.name == "posix":
            if not any(
                os.environ.get(v) for v in ("DISPLAY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS")
            ):
                return False
            if notify_send := shutil.which("notify-send"):
                res = subprocess.run(
                    [notify_send, "-a", APP_NAME, "-i", "dialog-warning", "-t", "5000", "--", title, message],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
                if res.returncode == 0:
                    return True
            if kdialog := shutil.which("kdialog"):
                res = subprocess.run(
                    [kdialog, "--passivepopup", message, "5", "--title", title],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
                if res.returncode == 0:
                    return True
            if zenity := shutil.which("zenity"):
                res = subprocess.run(
                    [zenity, "--notification", f"--text={title}: {message}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
                return res.returncode == 0
    except Exception:
        return False

    return False
