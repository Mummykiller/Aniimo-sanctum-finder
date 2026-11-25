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
set "DEPS_STAMP=%DIR%.venv\.deps_installed"

if exist "%LOG_FILE%" (
    for %%A in ("%LOG_FILE%") do if %%~zA GTR 1048576 move /y "%LOG_FILE%" "%LOG_FILE%.1" >nul 2>nul
)

set "NEED_SETUP=0"
if not exist "%DEPS_STAMP%" set "NEED_SETUP=1"
if not exist "%PYTHON_BIN%" set "NEED_SETUP=1"
if "%NEED_SETUP%"=="1" (
    if not exist "%PYTHON_BIN%" (
        python --version >nul 2>nul
        if errorlevel 1 (
            echo [%date% %time%] Python not found in PATH >> "%LOG_FILE%"
            exit /b 1
        )
        python -m venv "%DIR%.venv" >> "%LOG_FILE%" 2>&1
        if errorlevel 1 (
            rmdir /s /q "%DIR%.venv" 2>nul
            echo [%date% %time%] Failed to create .venv >> "%LOG_FILE%"
            exit /b 1
        )
    )
    "%PYTHON_BIN%" -m pip install -q -r "%DIR%requirements.txt" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo [%date% %time%] Failed to install dependencies >> "%LOG_FILE%"
        exit /b 1
    )
    type nul > "%DEPS_STAMP%"
)

"%PYTHON_BIN%" "%DIR%main.py" %* >> "%LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo [%date% %time%] launch.bat exited with code %RC% >> "%LOG_FILE%"
    exit /b %RC%
)
