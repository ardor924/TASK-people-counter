@echo off
setlocal

:: 1. 인코딩 및 타이틀 (ANSI로 저장 필수)
title AI People Counter - Final Launcher

echo ==================================================
echo [System] 환경 체크 및 가상환경 준비 중...
echo ==================================================

:: 2. Conda 감지
where conda >nul 2>nul
if %errorlevel% neq 0 goto VENV_MODE

:CONDA_MODE
echo [System] Conda 환경이 감지되었습니다.
:: 환경 생성 (이미 있으면 알아서 스킵됨)
call conda create -n ai_counter_env python=3.11 -y
:: 활성화 시도
call conda activate ai_counter_env
if %errorlevel% neq 0 (
    echo [Warning] Conda 활성화 실패. venv로 전환합니다.
    goto VENV_MODE
)
goto SETUP_CHECK

:VENV_MODE
echo [System] venv 방식을 사용합니다.
if not exist ".venv" (
    echo [System] .venv 생성 중...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] 가상환경 활성화 실패.
    pause
    exit
)

:SETUP_CHECK
echo [System] 라이브러리 체크 및 설치 시작 (setup.py)...
:: 가상환경이 활성화된 상태에서 setup.py 실행
:: (setup.py 내부에서 이미 설치 여부를 체크하므로 바로 호출해도 안전합니다)
python setup.py

echo.
echo ==================================================
echo [System] 시스템을 실행합니다. (main.py)
echo ==================================================
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 실행 중 문제가 발생했습니다.
    pause
)

endlocal