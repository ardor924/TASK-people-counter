import cv2
import numpy as np
import os
from collections import deque, Counter

class AIEngine:
    def __init__(self, model_path, line_pos=0.55):
        # 라이브러리 지연 로드
        from ultralytics import YOLO
        import supervision as sv
        self.sv = sv
        
        # 모델 로드
        if not os.path.exists(model_path): model_path = "yolov8n.pt"
        self.model = YOLO(model_path)
        
        # [ID 폭증 해결을 위한 끈질긴 트래커 설정]
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.2,   # 감지 문턱값 낮춤 (더 잘 잡게)
            lost_track_buffer=120,            # 4초간 사라져도 ID 유지 (끈질김)
            minimum_matching_threshold=0.5,   # [중요] 0.8->0.5 (모양이 좀 변해도 같은 ID로 인식)
            frame_rate=30
        )
        
        self.line_pos_ratio = line_pos
        self.line_zone = None 

    def init_line_zone(self, w, h):
        """ Supervision LineZone 정석 설정 """
        y = int(h * self.line_pos_ratio)
        start = self.sv.Point(0, y)
        end = self.sv.Point(w, y)
        
        # Supervision의 트리거 구역 생성
        self.line_zone = self.sv.LineZone(start=start, end=end)
        
        # 시각화용 선 객체 반환
        return self.line_zone

    def process_frame(self, frame):
        # 1. 추론
        results = self.model(frame, verbose=False)[0]
        detections = self.sv.Detections.from_ultralytics(results)
        
        # 2. 사람만 필터링 (class_id 0)
        detections = detections[detections.class_id == 0]

        # 3. 트래킹 (ID 부여)
        # 여기서 ID가 튀지 않도록 위에서 설정한 파라미터가 작동함
        detections = self.tracker.update_with_detections(detections)

        # 4. 카운팅 (Supervision LineZone Trigger)
        cross_in, cross_out = [], []
        
        # 감지된 객체가 있고, 라인존이 설정되어 있을 때만 트리거
        if self.line_zone is not None and len(detections) > 0:
            cross_in, cross_out = self.line_zone.trigger(detections=detections)
        else:
            # 객체가 없으면 False 리스트 생성
            cross_in = [False] * len(detections)
            cross_out = [False] * len(detections)
        
        return detections, cross_in, cross_out

    def get_counts(self):
        if self.line_zone:
            return self.line_zone.in_count, self.line_zone.out_count
        return 0, 0

class AttributeResolver:
    """ 속성 분석기 (Voting) """
    def __init__(self):
        self.votes = {}
        self.locked_info = {}
        # HSV 색상
        self.colors_hsv = {
            "Red":    ([0, 100, 100], [10, 255, 255]),
            "Blue":   ([90, 60, 40], [135, 255, 255]),
            "Green":  ([36, 50, 50], [85, 255, 255]),
            "Black":  ([0, 0, 0], [180, 255, 50]),
            "White":  ([0, 0, 200], [180, 20, 255]),
            "Gray":   ([0, 0, 50], [180, 50, 180])
        }

    def resolve(self, frame, xyxy, track_id):
        if track_id in self.locked_info:
            return self.locked_info[track_id]

        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return {"gender": "Unk", "g_conf": 0, "color": "Unk", "c_conf": 0}

        ch, cw = crop.shape[:2]
        upper = crop[int(ch*0.1):int(ch*0.5), int(cw*0.2):int(cw*0.8)]
        if upper.size == 0: upper = crop

        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        best_c = "Gray"
        max_p = 0
        for name, (l, u) in self.colors_hsv.items():
            mask = cv2.inRange(hsv, np.array(l), np.array(u))
            cnt = cv2.countNonZero(mask)
            if cnt > max_p:
                max_p = cnt
                best_c = name
        
        ratio = cw / ch if ch > 0 else 0
        gender = "Female" if ratio < 0.35 else "Male"

        if track_id not in self.votes:
            self.votes[track_id] = {'g': deque(maxlen=20), 'c': deque(maxlen=20)}
        self.votes[track_id]['g'].append(gender)
        self.votes[track_id]['c'].append(best_c)

        try:
            fg = Counter(self.votes[track_id]['g']).most_common(1)[0][0]
            fc = Counter(self.votes[track_id]['c']).most_common(1)[0][0]
            gc = Counter(self.votes[track_id]['g'])[fg] / len(self.votes[track_id]['g'])
            cc = Counter(self.votes[track_id]['c'])[fc] / len(self.votes[track_id]['c'])
        except:
            fg, gc = gender, 0.5
            fc, cc = best_c, 0.5
        
        if len(self.votes[track_id]['g']) > 15 and gc > 0.6:
            res = {"gender": fg, "g_conf": 0.99, "color": fc, "c_conf": 0.99}
            self.locked_info[track_id] = res
            return res
            
        return {"gender": fg, "g_conf": gc, "color": fc, "c_conf": cc}