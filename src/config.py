# src/config.py
import os

# =========================
# System Configurations
# =========================

# Paths
DEFAULT_VIDEO_PATH = "data/sample.avi"
# DEFAULT_VIDEO_PATH = "data/dev_day.mp4"
# DEFAULT_VIDEO_PATH = "data/eval_day.mp4"
# DEFAULT_VIDEO_PATH = "data/eval_night.mp4"
# DEFAULT_VIDEO_PATH = "data/eval_indoors.mp4"
MODEL_PATH = "models/yolov8n.pt"

# AI Parameters
RESIZE_WIDTH = 960
CONF_THRESH = 0.65       # 야간 오탐지 방지 임계값

# Tracking Parameters
DWELL_TIME_SEC = 0.5     # 카운팅 인정 체류 시간
FORGET_TIME_SEC = 0.7    # 트래킹 유실 허용 시간
TRACK_BUFFER_SEC = 2.0   # 트래커 버퍼
ID_FUSION_RATIO = 0.015  # ID 통합 거리 비율

# ROI Settings
ZONE_START_RATIO = 0.45  # 카운팅 라인 위치 (0.0 ~ 1.0)