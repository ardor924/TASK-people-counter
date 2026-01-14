import cv2

class PeopleCounter:
    def __init__(self, line_height_ratio=0.6):
        """
        line_height_ratio: 화면의 몇 % 지점에 선을 그을지 (0.6 = 위에서 60% 지점)
        """
        self.count = 0
        self.counted_ids = set() # 이미 센 사람 ID 저장 (중복 방지)
        self.line_ratio = line_height_ratio

    def update(self, center_y, total_height, track_id):
        """
        사람의 발(또는 중심)이 선을 넘었는지 확인
        """
        line_y = int(total_height * self.line_ratio)
        
        # 간단한 로직: 사람의 중심점이 선보다 아래에 있고(지나갔고), 아직 안 셌다면 카운트
        # (원래는 이전 프레임 위치와 비교해야 정확하지만, 약식으로 구현)
        if center_y > line_y and track_id not in self.counted_ids:
            self.count += 1
            self.counted_ids.add(track_id)
            return True # 방금 카운트 됨 (시각화용)
        
        return False

    def draw(self, frame):
        """ 화면에 카운팅 선과 숫자를 그림 """
        h, w = frame.shape[:2]
        line_y = int(h * self.line_ratio)
        
        # 1. 카운팅 라인 그리기 (노란색)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(frame, "Counting Line", (10, line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 2. 총 카운트 표시 (좌측 상단 박스)
        info_text = f"Total Count: {self.count}"
        cv2.rectangle(frame, (15, 15), (280, 60), (0, 0, 0), -1) # 검은 배경
        cv2.putText(frame, info_text, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)