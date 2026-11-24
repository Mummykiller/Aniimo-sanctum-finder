@echo off
setlocal
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "OPENCV_OPENCL_RUNTIME=disabled"

set "DIR=%~dp0"
set "LOG_FILE=%DIR%launcher.log"

if exist "%DIR%.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

"%PYTHON_BIN%" "%DIR%main.py" %* >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [%date% %time%] launch.bat exited with error >> "%LOG_FILE%"
    exit /b 1
)
