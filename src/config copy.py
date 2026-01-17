import os

# ==========================================
# 1. Path Configurations
# ==========================================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAMPLE_DIR = os.path.join(BASE_DIR, "best_samples")

# Default Video & Model Paths
DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, "sample.avi")
# DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, "dev_day.mp4")
# DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, "eval_day.mp4")
# DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, "eval_night.mp4")
# DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, "eval_indoors.mp4")
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")

# ==========================================
# 2. Detection & Tracking Parameters
# ==========================================
RESIZE_WIDTH = 960      # 영상 처리 해상도 (성능/정확도 균형)
CONF_THRESH = 0.65      # 탐지 신뢰도 임계값
TRACK_BUFFER_SEC = 2.0  # 트래킹 유지 시간 (초)

# ==========================================
# 3. Counting & Fusion Logic (ReID)
# ==========================================
ZONE_START_RATIO = 0.45 # ROI 판정선 위치 (화면 상단 기준 비율)
DWELL_TIME_SEC = 0.48   # 카운팅 인정 최소 체류 시간
FORGET_TIME_SEC = 0.7   # ID Fusion 유지 시간
ID_FUSION_RATIO = 0.02  # 동일인 판정 유클리드 거리 비율