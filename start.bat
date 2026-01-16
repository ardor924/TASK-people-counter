@echo off
setlocal

:: 1. 타이틀 설정
title AI People Counter - Final Launcher

echo ==================================================
echo [System] 시스템을 시작합니다.
echo ==================================================

:: 2. Conda 설치 경로 및 가상환경 파이썬 경로 직접 탐색
:: 사용자님의 아나콘다 기본 경로를 이용해 ai_counter_env 내부의 파이썬을 바로 찾습니다.
for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
set "PYTHON_EXE=%CONDA_BASE%\envs\ai_counter_env\python.exe"

:: 3. 만약 Conda 환경이 없다면 venv 체크
if not exist "%PYTHON_EXE%" (
    echo [System] Conda 환경을 찾을 수 없어 venv를 확인합니다.
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    ) else (
        echo [ERROR] 실행 가능한 가상환경이 없습니다. 
        echo 먼저 환경 구축을 완료해 주세요.
        pause
        exit /b
    )
)

echo [System] 사용 중인 파이썬: "%PYTHON_EXE%"

:: 4. setup.py 실행 (중복 설치 방지 로직 적용됨)
echo.
echo [System] 라이브러리 체크 중...
"%PYTHON_EXE%" setup.py

:: 5. main.py 실행
echo.
echo ==================================================
echo [System] 프로그램 실행 (main.py)
echo ==================================================
"%PYTHON_EXE%" main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 프로그램 실행 중 문제가 발생했습니다.
)

echo.
echo [System] 모든 프로세스가 종료되었습니다.
pause
endlocal