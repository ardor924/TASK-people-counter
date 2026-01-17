import os

# ==========================================
# 1. Path Configurations (파일 경로)
# ==========================================
# DEFAULT_VIDEO_PATH = "data/sample.avi"
# DEFAULT_VIDEO_PATH = "data/dev_day.mp4"
# DEFAULT_VIDEO_PATH = "data/eval_day.mp4"
# DEFAULT_VIDEO_PATH = "data/eval_night.mp4"
DEFAULT_VIDEO_PATH = "data/eval_indoors.mp4"

MODEL_PATH = "models/yolov8n.pt"

# ==========================================
# 2. AI Inference (추론 설정)
# ==========================================
RESIZE_WIDTH = 960      # 처리 해상도 (성능/정확도 밸런스)
CONF_THRESH = 0.65      # 탐지 임계값 (야간 오탐 방지)
# CONF_THRESH = 0.62    # 저조도 환경용

# ==========================================
# 3. Tracking & Re-ID (추적 및 재식별)
# ==========================================
DWELL_TIME_SEC = 0.48   # 카운팅 인정 최소 체류 시간
FORGET_TIME_SEC = 0.7   # 객체 유실 시 대기 시간
TRACK_BUFFER_SEC = 2.0  # 트래커 내부 버퍼 유지 시간
ID_FUSION_RATIO = 0.015 # 동일 객체 판정 거리 비율 (Re-ID 로직)

# ==========================================
# 4. ROI Settings (판정 구역)
# ==========================================
ZONE_START_RATIO = 0.45 # 카운팅 라인 위치 (화면 상단 기준 비율)