# Aniimo Sanctum Finder

A lightweight hotkey launcher for **Aniimo** that looks at your in-game Sanctum map and opens the matching layout page on `aniimotools.dev` in your default browser.

---

## How It Works (Plain English)

When you press your hotkey while viewing the in-game Sanctum map:

1. **Takes a quick snapshot of your main monitor only** (ignoring any second monitor).
2. **Finds the two doors:** It looks for the **orange** Main entrance door (or your yellow player arrow if you're standing on it) and the **blue** Side entrance door.
3. **Measures the starting room:** Using the starting room's dark outer walls as a ruler, it measures how far apart the two doors are, what direction the blue door is from the starting room, and which hallways (`North`, `East`, `South`, `West`) leave the starting room.
4. **Matches & opens the browser:** It compares those numbers against the lightweight `layouts.json` database and opens `https://aniimotools.dev/map/lost-sanctum/#plan-<id>` in your browser. If you press the hotkey outside the map screen, it exits quietly without opening anything.

> **Safety & Transparency:** This tool **never** reads game memory, **never** modifies game files, and **never** sends or injects inputs into the game.

---

## Installation

### Windows
1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) (check **"Add Python to PATH"** during install).
2. Download or clone this repository and double-click **`install_windows.bat`** inside `aniimo-sanctum-finder`.
   - It automatically creates `.venv`, installs the required packages, and creates an **Aniimo Sanctum Finder** shortcut on your Desktop.
   - *(Alternatively, running `launch.bat` directly will also auto-setup `.venv` on first launch.)*

### Linux (KDE Plasma Wayland / X11)
1. Ensure **Python 3.10+** is installed. On KDE Plasma Wayland, `spectacle` (installed by default on KDE) or `grim` is used for screen capture.
2. Open a terminal and run these commands step by step:
   ```bash
   git clone https://github.com/Mummykiller/Aniimo-sanctum-finder.git
   cd Aniimo-sanctum-finder
   chmod +x install_linux.sh
   ./install_linux.sh
   ```
   This sets up `.venv`, installs the required packages, marks `launch.sh` executable, and prints the exact path to paste into your KDE shortcut settings.

---

## How to Use

Open the Sanctum map in-game so both the starting room and the blue door are visible, then press your configured hotkey:

### Windows
1. Right-click the **Aniimo Sanctum Finder** shortcut on your Desktop (created by `install_windows.bat`).
2. Click **Properties** → click **Shortcut key** and press your desired hotkey (for example, `Ctrl+Alt+M`) → **OK**.

### Linux (KDE Plasma Wayland / X11)
1. Open **System Settings → Keyboard → Shortcuts → + Add New → Command or Script**.
2. Set the command to the full path printed by `./install_linux.sh`:
   ```text
   /path/to/aniimo-sanctum-finder/launch.sh
   ```
3. Click **Add**, select the new entry, click **+ Add custom shortcut**, press your desired key combination (for example, `Meta+M` or `Ctrl+Shift+X`), and click **Apply**.
4. Click out of System Settings (onto your desktop or game), open the Sanctum map, and press your hotkey!
   *(If KDE Plasma 6 doesn't trigger a newly added shortcut right away, log out and back in so `kwin_wayland` reloads it.)*

### Optional Command-Line Flags
- `--dry-run`: Prints the top 3 matched layouts and URL without opening your browser.
- `--calibrate`: Saves `calibrate_overlay.png` showing the detected doors, starting room square, corner checks, and hallway probes.
- `--test-image path/to/screenshot.png`: Tests detection against a saved screenshot file.

---

## Bug Reports & Troubleshooting

- Check `launcher.log` in the `aniimo-sanctum-finder` folder for error messages or match confidence details.
- If a layout is misidentified, run:
  ```bash
  .venv/bin/python main.py --test-image /path/to/screenshot.png --calibrate --dry-run
  ```
  and open a GitHub Issue with the layout number, `launcher.log` output, and `calibrate_overlay.png` (or a cropped screenshot of the map).

---

## License & Notices

- **License:** [MIT License](LICENSE)
- **AI-Assisted Disclosure & Non-Affiliation:** Created with AI assistance as a lightweight launcher for an external companion website (`aniimotools.dev`) that I do not own or operate. Not affiliated with or endorsed by that website or Pawprint Interactive.
- `©Pawprint Interactive Entertainment Pte. Ltd., fan-made, not official.`
