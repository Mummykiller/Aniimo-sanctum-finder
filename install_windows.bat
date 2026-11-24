@echo off
setlocal
set "DIR=%~dp0"
cd /d "%DIR%"

echo ============================================
echo  Aniimo Sanctum Finder - Windows Installer
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

if not exist "%DIR%.venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment in .venv...
    python -m venv "%DIR%.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment .venv already exists.
)

echo [2/3] Installing required packages...
"%DIR%.venv\Scripts\python.exe" -m pip install -q -r "%DIR%requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. Check your internet connection.
    pause
    exit /b 1
)

echo [3/3] Creating Desktop shortcut...
powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Aniimo Sanctum Finder.lnk'); $sc.TargetPath = '%DIR%launch.bat'; $sc.WorkingDirectory = '%DIR%'; $sc.WindowStyle = 7; $sc.Description = 'Aniimo Sanctum Finder'; $sc.Save()"

echo.
echo ============================================
echo  Installation complete!
echo ============================================
echo  1. Right-click "Aniimo Sanctum Finder" on your Desktop
echo  2. Click Properties ^> Shortcut key ^> press your hotkey (e.g. Ctrl+Alt+M)
echo  3. Click OK, open the Sanctum map in-game, and press your hotkey!
echo.
pause
