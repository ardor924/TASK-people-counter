@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI People Counter - Universal Launcher

echo ==================================================
echo [System] Initializing AI People Counter...
echo ==================================================

:: --------------------------------------------------------
:: [STEP 1] Determine Python Interpreter (TARGET_PY)
:: Priority: 1. Conda (ai_counter_env) -> 2. Local .venv
:: --------------------------------------------------------

set "TARGET_PY="
set "ENV_TYPE="

:: 1-1. Check if Conda is installed
where conda >nul 2>&1
if %errorlevel% equ 0 (
    :: Get Conda base path
    for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
    
    :: Check for specific user environment
    if exist "!CONDA_BASE!\envs\ai_counter_env\python.exe" (
        set "TARGET_PY=!CONDA_BASE!\envs\ai_counter_env\python.exe"
        set "ENV_TYPE=Conda (ai_counter_env)"
        goto :ENV_DECIDED
    )
)

:: 1-2. Check local .venv if Conda is missing or env not found
if exist ".venv\Scripts\python.exe" (
    set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
    set "ENV_TYPE=Local Venv (.venv)"
    goto :ENV_DECIDED
)

:: 1-3. No environment found. Preparing to create .venv
echo [System] No virtual environment found. Creating local '.venv'...

:: Find system Python (py or python)
set "SYS_PY="
where python >nul 2>&1 && set "SYS_PY=python"
if "%SYS_PY%"=="" (
    where py >nul 2>&1 && set "SYS_PY=py"
)

if "%SYS_PY%"=="" (
    echo.
    echo [ERROR] Python is not installed on this system.
    echo Please install Python 3.9 or higher and try again.
    pause
    exit /b
)

:: Create .venv
echo [System] Creating venv using Python (%SYS_PY%)...
%SYS_PY% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
set "ENV_TYPE=New Local Venv"

:ENV_DECIDED
echo [System] Detected Environment: %ENV_TYPE%
echo [System] Python Path: "%TARGET_PY%"


:: --------------------------------------------------------
:: [STEP 2] Install Libraries (Run Once)
:: --------------------------------------------------------

:: Check install completion flag
set "FLAG_FILE=.venv\install_complete.flag"

if "%ENV_TYPE%"=="Conda (ai_counter_env)" (
    :: For Conda, just check dependencies quickly via setup.py
    echo.
    echo [System] Checking library dependencies...
    "%TARGET_PY%" setup.py
) else (
    :: For Local venv, use flag file to skip checks
    if exist "%FLAG_FILE%" (
        echo [System] Installation skipped (Flag file found).
    ) else (
        echo.
        echo [System] Installing required libraries (First run only)...
        echo [System] Please wait...
        
        "%TARGET_PY%" -m pip install --upgrade pip
        "%TARGET_PY%" setup.py
        
        if !errorlevel! equ 0 (
            echo Done > "%FLAG_FILE%"
            echo [System] Installation complete.
        ) else (
            echo [ERROR] Error occurred during library installation.
            pause
            exit /b
        )
    )
)

:: --------------------------------------------------------
:: [STEP 3] Launch Main Program
:: --------------------------------------------------------
echo.
echo ==================================================
echo [System] Launching AI People Counter (VLM Mode)
echo ==================================================
echo.

"%TARGET_PY%" main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] An error occurred while running the program.
    pause
) else (
    echo.
    echo [System] Program finished successfully.
    :: Keep window open to see logs
    pause
)