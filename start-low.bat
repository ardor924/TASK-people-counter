@echo off
setlocal enabledelayedexpansion
:: 한글 깨짐 방지 및 UTF-8 설정
chcp 65001 >nul
cd /d "%~dp0"

title AI People Counter - Low-Spec Launcher

echo ==================================================
echo [System] Initializing AI People Counter (Low-Spec)...
echo ==================================================

:: --------------------------------------------------------
:: [STEP 1] Determine Python Interpreter (TARGET_PY)
:: --------------------------------------------------------

set "TARGET_PY="
set "ENV_TYPE="

:: 1-1. Check local .venv (가장 확실한 로컬 환경부터 체크)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
    set "ENV_TYPE=Local venv (.venv)"
    goto :ENV_DECIDED
)

:: 1-2. Check if Conda is installed
where conda >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
    if exist "!CONDA_BASE!\envs\ai_counter_env\python.exe" (
        set "TARGET_PY=!CONDA_BASE!\envs\ai_counter_env\python.exe"
        set "ENV_TYPE=Conda (ai_counter_env)"
        goto :ENV_DECIDED
    )
)

:: 1-3. Create new .venv if none exists
echo [System] No virtual environment found. Creating local '.venv'...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed.
    pause
    exit /b
)
python -m venv .venv
set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
set "ENV_TYPE=New Local Venv"

:ENV_DECIDED
echo [System] Detected Environment: %ENV_TYPE%
echo [System] Python Path: "%TARGET_PY%"

:: --------------------------------------------------------
:: [STEP 2] Install Libraries
:: --------------------------------------------------------

:: setup.py가 실제 존재하는지 확인 후 실행
if exist "setup.py" (
    if not exist ".venv\install_complete.flag" (
        echo [System] Installing libraries via setup.py...
        "%TARGET_PY%" setup.py
        echo done > ".venv\install_complete.flag"
    )
) else (
    echo [Warning] setup.py not found. Skipping auto-install.
)

:: --------------------------------------------------------
:: [STEP 3] Launch Main Program (Debug Mode)
:: --------------------------------------------------------
echo.
echo ==================================================
echo [System] Launching AI People Counter (Hybrid Mode)
echo ==================================================
echo.

:: 프로그램이 튕기는 원인을 보기 위해 python 대신 직접 실행하지 않고 호출
:: 영상 경로 등이 상대경로일 경우를 대비해 루트에서 실행 확인
"%TARGET_PY%" main-low.py

:: 에러 발생 시 창이 바로 닫히지 않게 강제 대기
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program crashed with exit code %errorlevel%.
    echo [Debug] Check if video file exists at: data/dev_day.mp4
    pause
) else (
    echo.
    echo [System] Program finished successfully.
    pause
)