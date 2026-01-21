@echo off
setlocal enabledelayedexpansion
:: 한글 출력 깨짐 방지
chcp 65001 >nul
cd /d "%~dp0"

title AI People Counter - Auto Launcher (Universal Safe Mode)

echo ==================================================
echo [System] Initializing AI People Counter...
echo ==================================================

:: --------------------------------------------------------
:: [STEP 1] Python Interpreter Selection & Creation
:: Priority: 1. Conda (Force Python 3.11) -> 2. Local .venv
:: --------------------------------------------------------

set "TARGET_PY="
set "ENV_TYPE="
set "CONDA_ENV_NAME=ai_counter_env"

:: 1-1. Check if Conda is available (Best Option)
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [System] Conda detected. Preparing environment '%CONDA_ENV_NAME%'...
    
    :: 1. 환경이 실제로 존재하는지 확인 (conda env list 활용)
    conda env list | findstr /C:"%CONDA_ENV_NAME% " >nul
    if !errorlevel! neq 0 (
        echo.
        echo [System] Creating new Conda env with Python 3.11...
        :: Base가 3.13이라도 여기서 3.11을 강제 지정하므로 안전함
        call conda create -n %CONDA_ENV_NAME% python=3.11 -y
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to create Conda environment.
            pause
            exit /b
        )
    )

    :: 2. [핵심 수정] 경로를 추측하지 않고, Conda에게 실행 파일 위치를 직접 물어봄
    :: 이 방식은 Base가 3.13이어도 3.11 환경의 경로를 정확히 가져옵니다.
    echo [System] Locating Python executable for %CONDA_ENV_NAME%...
    for /f "delims=" %%i in ('call conda run -n %CONDA_ENV_NAME% python -c "import sys; print(sys.executable)"') do set "TARGET_PY=%%i"

    set "ENV_TYPE=Conda (%CONDA_ENV_NAME% - Python 3.11)"
    goto :ENV_DECIDED
)

:: 1-2. Fallback to Local .venv
if exist "%~dp0.venv\Scripts\python.exe" (
    set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
    set "ENV_TYPE=Local Venv (.venv)"
    goto :ENV_DECIDED
)

:: 1-3. Create new local .venv (If Conda is missing)
echo [System] Conda not found. Creating local '.venv'...

set "SYS_PY="
where python >nul 2>&1 && set "SYS_PY=python"
if "%SYS_PY%"=="" (
    where py >nul 2>&1 && set "SYS_PY=py"
)

if "%SYS_PY%"=="" (
    echo.
    echo [ERROR] Python is not installed. Please install Python.
    pause
    exit /b
)

echo [System] Creating venv using system Python (%SYS_PY%)...
%SYS_PY% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
set "ENV_TYPE=New Local Venv (System Fallback)"

:ENV_DECIDED
echo [System] Detected Environment: %ENV_TYPE%
echo [System] Python Path: "%TARGET_PY%"

:: --------------------------------------------------------
:: [STEP 2] Install Libraries (Intelligent Logic)
:: --------------------------------------------------------

if exist "setup.py" (
    :: Conda 환경인 경우
    if "%ENV_TYPE:~0,5%"=="Conda" (
        echo.
        echo [System] Checking libraries in Conda env...
        "%TARGET_PY%" setup.py
    ) else (
        :: 로컬 venv인 경우 (버전 체크 로직)
        if not exist ".venv\install_complete.flag" (
            echo.
            echo [System] Installing libraries in local venv...
            
            "%TARGET_PY%" -m pip install --upgrade pip
            
            :: 파이썬 마이너 버전 확인 (예: 3.13 -> 13) 
            for /f "delims=" %%v in ('"%TARGET_PY%" -c "import sys; print(sys.version_info.minor)"') do set PY_MINOR=%%v
            
            :: 3.13 이상이면 CPU 강제 설치
            if !PY_MINOR! GEQ 13 (
                echo.
                echo [Warning] Python 3.13+ detected.
                echo [System] Forcing CPU version of PyTorch for compatibility...
                "%TARGET_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ) else (
                echo.
                echo [System] Python 3.!PY_MINOR! detected.
                echo [System] Installing standard PyTorch...
            )
            
            "%TARGET_PY%" setup.py
            echo Done > ".venv\install_complete.flag"
        )
    )
)

:: --------------------------------------------------------
:: [STEP 3] Launch Main Program
:: --------------------------------------------------------
echo.
echo ==================================================
echo [System] Launching AI People Counter
echo ==================================================
echo.

:: start-low.bat 은 여기를 main-low.py로 변경
"%TARGET_PY%" main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program crashed with exit code %errorlevel%.
    pause
) else (
    echo.
    echo [System] Program finished successfully.
    pause
)