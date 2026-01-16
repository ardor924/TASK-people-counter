@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI People Counter - Universal Launcher

echo ==================================================
echo [System] AI People Counter 실행 준비 중...
echo ==================================================

:: --------------------------------------------------------
:: [STEP 1] 실행할 파이썬(TARGET_PY) 결정 로직
:: 우선순위: 1. Conda 환경 (ai_counter_env) -> 2. 로컬 .venv
:: --------------------------------------------------------

set "TARGET_PY="
set "ENV_TYPE="

:: 1-1. Conda 감지 시도 (에러 방지: where 명령어로 존재 여부 먼저 확인)
where conda >nul 2>&1
if %errorlevel% equ 0 (
    :: Conda가 설치된 경우에만 실행
    for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
    
    :: 사용자의 특정 환경 경로 체크
    if exist "!CONDA_BASE!\envs\ai_counter_env\python.exe" (
        set "TARGET_PY=!CONDA_BASE!\envs\ai_counter_env\python.exe"
        set "ENV_TYPE=Conda (ai_counter_env)"
        goto :ENV_DECIDED
    )
)

:: 1-2. Conda가 없거나 해당 환경이 없으면 -> 로컬 .venv 확인
if exist ".venv\Scripts\python.exe" (
    set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
    set "ENV_TYPE=Local Venv (.venv)"
    goto :ENV_DECIDED
)

:: 1-3. 아무것도 없다면 -> 시스템 파이썬 찾아서 .venv 생성 준비
echo [System] 설정된 가상환경이 없습니다. 로컬 환경(.venv)을 생성합니다.

:: 시스템 파이썬 찾기 (py 또는 python)
set "SYS_PY="
where python >nul 2>&1 && set "SYS_PY=python"
if "%SYS_PY%"=="" (
    where py >nul 2>&1 && set "SYS_PY=py"
)

if "%SYS_PY%"=="" (
    echo.
    echo [ERROR] 시스템에 Python이 설치되어 있지 않습니다.
    echo Python 3.9 이상을 설치하고 다시 실행해주세요.
    pause
    exit /b
)

:: .venv 생성
echo [System] Python(%SYS_PY%)을 사용하여 가상환경 생성 중...
%SYS_PY% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] 가상환경 생성 실패.
    pause
    exit /b
)

set "TARGET_PY=%~dp0.venv\Scripts\python.exe"
set "ENV_TYPE=New Local Venv"

:ENV_DECIDED
echo [System] 감지된 환경: %ENV_TYPE%
echo [System] Python 경로: "%TARGET_PY%"


:: --------------------------------------------------------
:: [STEP 2] 라이브러리 설치 (최초 1회만 수행)
:: --------------------------------------------------------

:: 설치 완료 플래그 파일 확인
set "FLAG_FILE=.venv\install_complete.flag"
if "%ENV_TYPE%"=="Conda (ai_counter_env)" (
    :: Conda 환경일 경우 플래그 파일 위치 조정 (혹은 setup.py를 매번 빠르게 체크)
    :: 편의상 Conda는 이미 세팅되었다고 가정하되, 안전을 위해 setup.py 호출 (빠름)
    echo.
    echo [System] 라이브러리 의존성을 체크합니다...
    "%TARGET_PY%" setup.py
) else (
    :: 로컬 venv인 경우 플래그 파일로 스킵 처리
    if exist "%FLAG_FILE%" (
        echo [System] 기존 설치 내역이 확인되어 설치를 건너뜁니다.
    ) else (
        echo.
        echo [System] 필수 라이브러리를 설치합니다. (최초 1회)
        echo [System] 잠시만 기다려 주세요...
        
        "%TARGET_PY%" -m pip install --upgrade pip
        "%TARGET_PY%" setup.py
        
        if !errorlevel! equ 0 (
            echo Done > "%FLAG_FILE%"
            echo [System] 설치 완료.
        ) else (
            echo [ERROR] 라이브러리 설치 중 오류 발생.
            pause
            exit /b
        )
    )
)

:: --------------------------------------------------------
:: [STEP 3] 메인 프로그램 실행
:: --------------------------------------------------------
echo.
echo ==================================================
echo [System] AI People Counter 실행 (VLM Mode)
echo ==================================================
echo.

"%TARGET_PY%" main-low.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 프로그램 실행 중 오류가 발생했습니다.
    pause
) else (
    echo.
    echo [System] 프로그램이 정상적으로 종료되었습니다.
    :: 창이 바로 꺼지는 것을 원치 않으면 아래 pause 유지
    pause
)