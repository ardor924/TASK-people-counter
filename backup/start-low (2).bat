@echo off
setlocal

:: 1. 타이틀 설정 (저사양 모드)
title AI People Counter - Low Spec Mode

echo ==================================================
echo [System] 저사양(CPU/내장그래픽) 모드로 시작합니다.
echo ==================================================

:: 2. Conda 설치 경로 탐색 (여러 경로를 순차적으로 확인)
set "CONDA_BASE="
for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set "CONDA_BASE=%%i"

:: 만약 conda 명령어가 직접 안 먹히면 기본 사용자 경로 확인
if "%CONDA_BASE%"=="" (
    if exist "%USERPROFILE%\anaconda3" set "CONDA_BASE=%USERPROFILE%\anaconda3"
    if exist "%USERPROFILE%\Miniconda3" set "CONDA_BASE=%USERPROFILE%\Miniconda3"
)

:: 3. Python 실행 파일 경로 확정
set "PYTHON_EXE=%CONDA_BASE%\envs\ai_counter_env\python.exe"

:: 4. 가상환경 존재 여부 체크
if not exist "%PYTHON_EXE%" (
    echo [System] Conda 환경을 찾을 수 없어 venv를 확인합니다.
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    ) else (
        echo.
        echo [ERROR] 실행 가능한 가상환경(ai_counter_env 또는 .venv)이 없습니다!
        echo [Solution] start.bat을 먼저 실행하여 환경 구축을 완료해 주세요.
        echo.
        pause
        exit /b
    )
)

echo [System] 사용 중인 파이썬: "%PYTHON_EXE%"

:: 5. setup.py 실행
echo.
echo [System] 라이브러리 및 모델 체크 중...
"%PYTHON_EXE%" setup.py
if %errorlevel% neq 0 (
    echo [Warning] setup.py 실행 중 경고가 발생했습니다.
)

:: 6. main_low.py 실행
echo.
echo ==================================================
echo [System] 프로그램 실행 (main_low.py)
echo ==================================================
"%PYTHON_EXE%" main_low.py

:: 7. 튕김 방지 (에러 발생 시 메시지 확인용)
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] main_low.py 실행 중 치명적인 오류가 발생했습니다.
    echo (에러 메시지를 확인하신 후 아무 키나 누르세요.)
    pause
) else (
    echo.
    echo [System] 분석이 정상적으로 종료되었습니다.
    pause
)

endlocal