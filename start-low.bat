@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI People Counter - Low-Spec Launcher

echo ==================================================
echo [System] Initializing AI People Counter (Low-Spec)...
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
    for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
    
    if exist "!CONDA_BASE!\envs\ai_counter_env\python.exe" (
        set "TARGET_PY=!CONDA_BASE!\envs\ai_counter_env\python.exe"
        set "ENV_TYPE=Conda (ai_counter_env)"
        goto :ENV_DECIDED
    )
)

:: 1-2. Check local .venv
if exist ".venv\Scripts\python.exe" (
    set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
    set "ENV_TYPE=Local venv (.venv)"
    goto :ENV_DECIDED
)

:: 1-3. Create new .venv if none exists
echo [System] No virtual environment found. Creating local '.venv'...

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

set "FLAG_FILE=.venv\install_complete.flag"

if "%ENV_TYPE%"=="Conda (ai_counter_env)" (
    echo.
    echo [System] Checking library dependencies...
    "%TARGET_PY%" setup.py
) else (
    if exist "%FLAG_FILE%" (
        echo [System] Skipping installation (Already installed).
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
:: [STEP 3] Launch Main Program (Low-Spec Mode)
:: --------------------------------------------------------
echo.
echo ==================================================
echo [System] Launching AI People Counter (Hybrid Mode)
echo ==================================================
echo.

"%TARGET_PY%" main-low.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] An error occurred while running the program.
    pause
) else (
    echo.
    echo [System] Program finished successfully.
    pause
)