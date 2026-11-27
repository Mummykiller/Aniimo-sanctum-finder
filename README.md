# Aniimo Sanctum Finder

A lightweight hotkey launcher for **Aniimo** that snapshots your in-game Sanctum map and opens the matching layout on `aniimotools.dev` in your default browser.

> **Safety:** Never reads game memory, never modifies game files, and never injects inputs into the game.

---

## How It Works

1. **Captures your primary monitor** (ignoring secondary monitors).
2. **Finds both doors:** Detects the **orange** Main entrance and **blue** Side entrance doors, automatically handling either starting room (orange or blue).
3. **Built-in fallbacks:**
   - **Player arrow anchor:** If your yellow player arrow is standing on top of a door and hides it, the finder uses the arrow as a fallback anchor and shows a warning notification that the match may be less accurate.
   - **Edge retry:** If a door is near the edge of the screen and clipped by the default 5% border trim, it automatically retries with no margin trim.
4. **Matches & notifies:** Compares door distance, angle, and starting-room exits against `layouts.json` and opens `https://aniimotools.dev/map/lost-sanctum/#plan-<id>`.
   - **Clean match (90%+):** Opens the map silently.
   - **Uncertain / partial capture:** Sends a native desktop notification if confidence is below `90%`, if the player-arrow fallback was used, or if only one door is visible (so you know to zoom out). If you press the hotkey outside the map screen, it exits quietly.

---

## Setup & Usage (Windows / Linux)

<details>
<summary><strong>Windows</strong> (Install, Hotkey & Notifications)</summary>

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) (check **"Add Python to PATH"**).
2. Download and **Extract All** from the ZIP (don't run from inside the ZIP preview), then double-click **`install_windows.bat`** inside `Aniimo-sanctum-finder`.
   - Creates `.venv`, installs dependencies, and places a silent **Aniimo Sanctum Finder** shortcut (`pythonw.exe`, no console flash) on your Desktop.
   - *(Running `launch.bat` directly also auto-sets up `.venv` on first launch.)*
3. **Bind a hotkey:**
   - Right-click **Aniimo Sanctum Finder** on your Desktop → **Properties** → **Shortcut key** → press your hotkey (e.g. `Ctrl+Alt+M`) → **OK**.
   - *(Keep the shortcut on your Desktop; moving or deleting it removes the hotkey.)*
4. **In-game settings:** Use **Borderless Windowed** mode (Exclusive Fullscreen captures a black screen on Windows). If Windows **HDR / Auto-HDR** is on and colors look washed out, disable HDR for accurate door detection. If Aniimo runs as Administrator, Windows blocks unelevated hotkeys while the game has focus.
5. **Use it:** Open the Sanctum map in-game so both doors are visible and press your hotkey.
6. **Notifications:** Uses built-in Windows PowerShell Toast / tray notifications (no extra setup required; Windows Focus Assist / Do Not Disturb may mute toasts during gameplay).

</details>

<details>
<summary><strong>Linux</strong> (Install, Hotkey & Notifications)</summary>

1. Ensure **Python 3.10+** is installed. On KDE Plasma Wayland, `spectacle` (default on KDE) or `grim` is used for screen capture.
2. Run the installer:
   ```bash
   git clone https://github.com/Mummykiller/Aniimo-sanctum-finder.git
   cd Aniimo-sanctum-finder
   chmod +x install_linux.sh
   ./install_linux.sh
   ```
   This creates `.venv`, installs dependencies, marks `launch.sh` executable, and prints the exact path for your shortcut.
3. **Bind a hotkey (KDE Plasma Wayland / X11):**
   - Open **System Settings → Keyboard → Shortcuts → + Add New → Command or Script**.
   - Paste `/path/to/Aniimo-sanctum-finder/launch.sh`.
   - Click **Add**, assign your shortcut (e.g. `Meta+M` or `Ctrl+Shift+X`), and click **Apply**.
   - *(On KDE Plasma 6, log out and back in if `kwin_wayland` doesn't pick up a new shortcut immediately.)*
4. **Use it:** Open the Sanctum map in-game so both doors are visible and press your hotkey.
5. **Notifications:** Uses `notify-send` (with `kdialog` / `zenity` fallbacks) for low-confidence or partial-door warnings.

</details>

---

## CLI Flags & Troubleshooting

* `--dry-run`: Prints top 3 candidates and URL without opening the browser or sending desktop notifications.
* `--calibrate`: Saves `calibrate_overlay.png` with detected doors, room box, and exit probes.
* `--test-image <path>`: Runs detection and matching on a saved screenshot.
* `--no-notify`: Disables desktop notifications.
* `--no-sync`: Skips checking GitHub for `layouts.json` updates.
* `--min-confidence <0..1>`: Sets the minimum match threshold to open a URL (default `0.65`).

If a layout is misidentified, check `launcher.log` or run:
```bash
# Linux
.venv/bin/python main.py --test-image /path/to/screenshot.png --calibrate --dry-run

# Windows
.venv\Scripts\python.exe main.py --test-image C:\path\to\screenshot.png --calibrate --dry-run
```
and open a GitHub Issue with the layout number, `launcher.log` output, and `calibrate_overlay.png`.

---

## License & Notices

* **License:** [MIT License](LICENSE)
* **AI-Assisted Disclosure & Non-Affiliation:** Created with AI assistance as a lightweight launcher for `aniimotools.dev`, which I do not own or operate. Not affiliated with or endorsed by that website or Pawprint Interactive.
* `©Pawprint Interactive Entertainment Pte. Ltd., fan-made, not official.`
