from ultralytics import YOLO

class PeopleDetector:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
        
        # [핵심] 트래커 설정 튜닝
        # track_buffer: 사라진 객체를 기억하는 프레임 수 (기본 30 -> 60으로 증가)
        # match_thresh: 매칭 임계값 (기본 0.5 -> 0.8로 높여서 확실한 것만 매칭)
        self.tracker_args = {
            "tracker": "bytetrack.yaml", 
            "track_buffer": 60,  # 2초 동안 사라져도 기억함 (ID 유지력 상승)
            "match_thresh": 0.8
        }

    def track(self, frame):
        """
        [수정됨] persist=True 추가
        이 옵션이 없으면 매 프레임을 독립적으로 봐서 ID가 끊깁니다.
        """
        results = self.model.track(
            frame, 
            persist=True,               # [중요] ID 유지를 위한 필수 옵션
            classes=[0],                # 사람(Class 0)만 추적
            conf=0.3,                   # 감지 신뢰도 (너무 낮으면 노이즈 잡힘)
            iou=0.5,                    # NMS 임계값
            tracker="bytetrack.yaml",   # ByteTrack 사용
            verbose=False
        )
        return results[0]