from ultralytics import YOLO
import os

class PeopleDetector:
    def __init__(self, model_name='models/yolov8n.pt'):
        # 경로 예외 처리만 유지
        if not os.path.exists(model_name):
            model_name = 'yolov8n.pt'
        self.model = YOLO(model_name)
        
    def track(self, frame):
        # 외부 설정 파일 로드
        tracker_config = "custom_tracker.yaml"
        if not os.path.exists(tracker_config):
            tracker_config = "bytetrack.yaml"

        # 순정 트래킹 실행
        results = self.model.track(
            frame, 
            persist=True,
            classes=[0],        # Person Only
            conf=0.25,          # 표준 신뢰도
            iou=0.5,            # 표준 IOU
            tracker=tracker_config,
            verbose=False
        )
        return results[0]