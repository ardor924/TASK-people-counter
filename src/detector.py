import os
import torch  # torch 추가
from ultralytics import YOLO

class PeopleDetector:
    def __init__(self, model_name='yolov8n.pt'):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, 'models', model_name)
        
        # GPU 사용 가능 여부 확인
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Device Setting: {self.device} (cuda=GPU, cpu=Slow)")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model = YOLO(self.model_path) 

    def track(self, frame):
        # device=self.device 추가 (여기가 핵심)
        # half=True: FP16 연산 사용 (정확도 거의 동일, 속도 2배 향상)
        results = self.model.track(frame, persist=True, classes=[0], 
                                   verbose=False, conf=0.5, iou=0.5, 
                                   device=self.device, half=True)
        return results[0]