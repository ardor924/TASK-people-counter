@echo off
chcp 65001 > nul
title AI People Counter - Low Spec (Isolated)

echo ==================================================
echo [System] 저사양 모드 환경 체크 중...
echo ==================================================

if not exist ".venv" (
    echo [System] 가상환경 생성 중...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python -c "import torch" 2> nul
if %errorlevel% neq 0 (
    echo [System] 환경 설정 시작 (setup.py 호출)...
    python setup.py
)

echo.
echo ==================================================
echo [System] 저사양 경량화 시스템 실행 (Isolated Mode)
echo ==================================================
python main_low.py
if %errorlevel% neq 0 pause