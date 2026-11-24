@echo off
setlocal
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "OPENCV_OPENCL_RUNTIME=disabled"

set "DIR=%~dp0"
set "LOG_FILE=%DIR%launcher.log"
set "PYTHON_BIN=%DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_BIN%" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [%date% %time%] Python not found in PATH >> "%LOG_FILE%"
        exit /b 1
    )
    python -m venv "%DIR%.venv" >> "%LOG_FILE%" 2>&1
    "%PYTHON_BIN%" -m pip install -q -r "%DIR%requirements.txt" >> "%LOG_FILE%" 2>&1
)

"%PYTHON_BIN%" "%DIR%main.py" %* >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [%date% %time%] launch.bat exited with error >> "%LOG_FILE%"
    exit /b 1
)
