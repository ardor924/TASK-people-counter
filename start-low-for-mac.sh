#!/bin/bash

echo "=================================================="
echo "[System] AI People Counter - Low Spec Mode (Mac/Linux)"
echo "=================================================="

# 1. Python 실행 파일 찾기 (Conda -> venv 순서)
# Conda 경로 추적
CONDA_BASE=$(conda info --base 2>/dev/null)
PYTHON_EXE="$CONDA_BASE/envs/ai_counter_env/bin/python"

if [ -f "$PYTHON_EXE" ]; then
    echo "[System] Conda 환경 감지됨: $PYTHON_EXE"
else
    # Conda 없으면 venv 확인
    if [ -f ".venv/bin/python" ]; then
        PYTHON_EXE=".venv/bin/python"
        echo "[System] venv 환경 감지됨: $PYTHON_EXE"
    else
        echo "[ERROR] 가상환경을 찾을 수 없습니다."
        echo "먼저 python 3.11 환경을 생성하거나 start.bat(Windows)을 참고하세요."
        exit 1
    fi
fi

# 2. setup.py 실행
echo ""
echo "[System] 라이브러리 및 모델 체크 중..."
"$PYTHON_EXE" setup.py

# 3. main_low.py 실행
echo ""
echo "=================================================="
echo "[System] 프로그램 실행 (main_low.py)"
echo "=================================================="
"$PYTHON_EXE" main_low.py