#!/bin/bash

# 창 제목 설정
echo -ne "\033]0;AI People Counter - macOS (Isolated)\007"

echo "=================================================="
echo "[System] Checking Isolated Environment (.venv)..."
echo "=================================================="

# 1. 가상환경 생성
if [ ! -d ".venv" ]; then
    echo "[System] Creating virtual environment..."
    python3 -m venv .venv
fi

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. 환경 검사 및 설치
python3 -c "import torch" 2> /dev/null
if [ $? -ne 0 ]; then
    echo "[System] Installing dependencies via setup.py..."
    python3 setup.py
fi

echo ""
echo "=================================================="
echo "[System] Launching Low-Spec System on macOS"
echo "=================================================="
python3 main_low.py

echo "Press enter to exit..."
read