import cv2

class PeopleCounter:
    def __init__(self, line_height_ratio=0.55):
        self.count = 0
        self.line_height_ratio = line_height_ratio # 화면의 55% 지점에 라인 생성
        self.counted_ids = set()
        self.previous_positions = {} # {track_id: y_center}

    def update(self, current_y, frame_height, track_id):
        line_y = int(frame_height * self.line_height_ratio)
        
        # 이미 센 사람은 패스
        if track_id in self.counted_ids:
            self.previous_positions[track_id] = current_y
            return False

        # 이전 위치가 없으면 현재 위치 저장하고 리턴
        if track_id not in self.previous_positions:
            self.previous_positions[track_id] = current_y
            return False

        prev_y = self.previous_positions[track_id]
        
        # [카운팅 로직] 
        # 이전에는 선 위에 있었는데(prev_y < line), 지금은 선 아래(current_y >= line)면 통과
        # 혹은 그 반대도 포함 (양방향 카운팅이 안전함)
        if (prev_y < line_y <= current_y) or (prev_y > line_y >= current_y):
            self.count += 1
            self.counted_ids.add(track_id)
            self.previous_positions[track_id] = current_y
            return True # 카운트 성공 신호
        
        self.previous_positions[track_id] = current_y
        return False

    def draw(self, frame):
        h, w = frame.shape[:2]
        line_y = int(h * self.line_height_ratio)
        
        # 카운팅 라인 그리기 (노란색)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        
        # 카운트 정보 표시 박스
        cv2.rectangle(frame, (15, 15), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Count: {self.count}", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)