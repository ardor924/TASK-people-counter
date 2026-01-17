# src/config.py
import os

# ==========================================
# 1. Path Configurations (파일 경로)
# ==========================================
# [사용자 설정] 테스트하고 싶은 영상 경로를 주석 해제하세요.
# DEFAULT_VIDEO_PATH = "data/dev_day.mp4"
# DEFAULT_VIDEO_PATH = "data/eval_day.mp4"
# DEFAULT_VIDEO_PATH = "data/eval_night.mp4"
DEFAULT_VIDEO_PATH = "data/eval_indoors4445.mp4"

# [시스템 안전장치] 위 파일이 없을 경우 강제로 실행할 보장된 샘플 경로
FALLBACK_VIDEO_PATH = "data/sample.avi"

MODEL_PATH = "models/yolov8n.pt"

# ==========================================
# 2. AI Inference (추론 설정)
# ==========================================
RESIZE_WIDTH = 960      # 처리 해상도
CONF_THRESH = 0.65      # 탐지 임계값

# ==========================================
# 3. Tracking & Re-ID (추적 및 재식별)
# ==========================================
DWELL_TIME_SEC = 0.48   
FORGET_TIME_SEC = 0.7   
TRACK_BUFFER_SEC = 2.0  
ID_FUSION_RATIO = 0.015 

# ==========================================
# 4. ROI Settings (판정 구역)
# ==========================================
ZONE_START_RATIO = 0.45