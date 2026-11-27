@echo off
setlocal EnableExtensions
set "DIR=%~dp0"
cd /d "%DIR%"

echo ============================================
echo  Aniimo Sanctum Finder - Windows Installer
echo ============================================
echo.

set "PY_CMD="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=py -3"
    goto :found_python
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :found_python
)
python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=python3"
    goto :found_python
)

echo [ERROR] Python 3.10+ is not installed or not working properly.
echo Note: Windows includes a fake python.exe shortcut that opens Microsoft Store.
echo Please install Python 3.10+ from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found_python
echo Using Python: %PY_CMD%
%PY_CMD% --version
echo.

if exist "%DIR%.venv" if not exist "%DIR%.venv\Scripts\python.exe" (
    echo [0/3] Removing incompatible leftover .venv folder...
    rmdir /s /q "%DIR%.venv" 2>nul
)

if not exist "%DIR%.venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment in .venv...
    %PY_CMD% -m venv "%DIR%.venv"
    if errorlevel 1 (
        rmdir /s /q "%DIR%.venv" 2>nul
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
type nul > "%DIR%.venv\.deps_installed"

echo [3/3] Creating Desktop shortcut (silent, no console flash)...
set "PS_BIN=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_BIN%" set "PS_BIN=powershell"
"%PS_BIN%" -NoProfile -Command "$d = $env:DIR; $wd = if ($d.Length -gt 3) { $d.TrimEnd('\') } else { $d }; $pyw = Join-Path $d '.venv\Scripts\pythonw.exe'; $main = Join-Path $d 'main.py'; $ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Aniimo Sanctum Finder.lnk')); if (Test-Path $pyw) { $sc.TargetPath = $pyw; $sc.Arguments = '\"' + $main + '\"'; $sc.WindowStyle = 1 } else { $sc.TargetPath = Join-Path $d 'launch.bat'; $sc.WindowStyle = 7 }; $sc.WorkingDirectory = $wd; $sc.Description = 'Aniimo Sanctum Finder'; $sc.Save()"
if errorlevel 1 (
    echo [ERROR] Failed to create Desktop shortcut.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Installation complete!
echo ============================================
echo  1. Right-click "Aniimo Sanctum Finder" on your Desktop
echo  2. Click Properties ^> Shortcut key ^> press your hotkey (e.g. Ctrl+Alt+M)
echo  3. Click OK, switch Aniimo to Borderless Windowed mode, open the map, and press your hotkey!
echo.
pause
