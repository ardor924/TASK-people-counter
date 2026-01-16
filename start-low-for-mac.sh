#!/bin/bash

# ==================================================
# AI People Counter - Universal Launcher (Mac/Linux)
# ==================================================

# 스크립트가 있는 디렉토리로 이동 (경로 문제 방지)
cd "$(dirname "$0")"

echo "=================================================="
echo "[System] AI People Counter (Low-Spec) 실행 준비 중..."
echo "=================================================="

# --------------------------------------------------------
# [STEP 1] 실행할 파이썬(TARGET_PY) 결정 로직
# 우선순위: 1. Conda 환경 (ai_counter_env) -> 2. 로컬 .venv
# --------------------------------------------------------

TARGET_PY=""
ENV_TYPE=""

# 1-1. Conda 감지 시도
if command -v conda &> /dev/null; then
    # Conda 기본 경로 확인
    CONDA_BASE=$(conda info --base)
    
    # ai_counter_env 환경이 있는지 확인
    if [ -f "$CONDA_BASE/envs/ai_counter_env/bin/python" ]; then
        TARGET_PY="$CONDA_BASE/envs/ai_counter_env/bin/python"
        ENV_TYPE="Conda (ai_counter_env)"
    fi
fi

# 1-2. Conda가 없거나 해당 환경이 없으면 -> 로컬 .venv 확인
if [ -z "$TARGET_PY" ] && [ -f ".venv/bin/python" ]; then
    TARGET_PY="./.venv/bin/python"
    ENV_TYPE="Local Venv (.venv)"
fi

# 1-3. 아무것도 없다면 -> 시스템 파이썬 찾아서 .venv 생성 준비
if [ -z "$TARGET_PY" ]; then
    echo "[System] 설정된 가상환경이 없습니다. 로컬 환경(.venv)을 생성합니다."

    # 시스템 파이썬 찾기 (python3 우선)
    if command -v python3 &> /dev/null; then
        SYS_PY="python3"
    elif command -v python &> /dev/null; then
        SYS_PY="python"
    else
        echo ""
        echo "[ERROR] 시스템에 Python이 설치되어 있지 않습니다."
        echo "Python 3.9 이상을 설치하고 다시 실행해주세요."
        exit 1
    fi

    # .venv 생성
    echo "[System] Python($SYS_PY)을 사용하여 가상환경 생성 중..."
    $SYS_PY -m venv .venv
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] 가상환경 생성 실패."
        exit 1
    fi

    TARGET_PY="./.venv/bin/python"
    ENV_TYPE="New Local Venv"
fi

echo "[System] 감지된 환경: $ENV_TYPE"
echo "[System] Python 경로: $TARGET_PY"


# --------------------------------------------------------
# [STEP 2] 라이브러리 설치 (최초 1회만 수행)
# --------------------------------------------------------

FLAG_FILE=".venv/install_complete.flag"

if [ "$ENV_TYPE" == "Conda (ai_counter_env)" ]; then
    # Conda 환경은 setup.py를 빠르게 체크
    echo ""
    echo "[System] 라이브러리 의존성을 체크합니다..."
    "$TARGET_PY" setup.py
else
    # 로컬 venv인 경우 플래그 파일 확인
    if [ -f "$FLAG_FILE" ]; then
        echo "[System] 기존 설치 내역이 확인되어 설치를 건너뜁니다."
    else
        echo ""
        echo "[System] 필수 라이브러리를 설치합니다. (최초 1회)"
        echo "[System] 잠시만 기다려 주세요..."
        
        "$TARGET_PY" -m pip install --upgrade pip
        "$TARGET_PY" setup.py
        
        if [ $? -eq 0 ]; then
            touch "$FLAG_FILE"
            echo "[System] 설치 완료."
        else
            echo "[ERROR] 라이브러리 설치 중 오류 발생."
            exit 1
        fi
    fi
fi

# --------------------------------------------------------
# [STEP 3] 메인 프로그램 실행 (main-low.py)
# --------------------------------------------------------
echo ""
echo "=================================================="
echo "[System] 프로그램 실행 (main-low.py)"
echo "=================================================="
echo ""

"$TARGET_PY" main-low.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] 프로그램 실행 중 오류가 발생했습니다."
    read -p "Press any key to exit..."
else
    echo ""
    echo "[System] 프로그램이 정상적으로 종료되었습니다."
fi