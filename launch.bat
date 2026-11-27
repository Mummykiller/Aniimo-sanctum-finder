@echo off
setlocal EnableExtensions
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "OPENCV_OPENCL_RUNTIME=disabled"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "DIR=%~dp0"
set "LOG_FILE=%DIR%launcher.log"
set "PYTHON_BIN=%DIR%.venv\Scripts\python.exe"
set "PYTHONW_BIN=%DIR%.venv\Scripts\pythonw.exe"
set "DEPS_STAMP=%DIR%.venv\.deps_installed"

if exist "%DEPS_STAMP%" if exist "%PYTHON_BIN%" goto :run_app

if exist "%DIR%.venv" if not exist "%PYTHON_BIN%" (
    rmdir /s /q "%DIR%.venv" 2>nul
)

if exist "%PYTHON_BIN%" goto :install_deps

set "PY_CMD="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=py -3"
    goto :create_venv
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :create_venv
)
python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=python3"
    goto :create_venv
)

echo [%date% %time%] Python 3.10+ not found in PATH >> "%LOG_FILE%"
exit /b 1

:create_venv
%PY_CMD% -m venv "%DIR%.venv" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    rmdir /s /q "%DIR%.venv" 2>nul
    echo [%date% %time%] Failed to create .venv >> "%LOG_FILE%"
    exit /b 1
)

:install_deps
"%PYTHON_BIN%" -m pip install -q -r "%DIR%requirements.txt" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [%date% %time%] Failed to install dependencies >> "%LOG_FILE%"
    exit /b 1
)
type nul > "%DEPS_STAMP%"

:run_app
if "%~1"=="" if exist "%PYTHONW_BIN%" (
    start "" "%PYTHONW_BIN%" "%DIR%main.py"
    exit /b 0
)
"%PYTHON_BIN%" "%DIR%main.py" %*
exit /b %ERRORLEVEL%
