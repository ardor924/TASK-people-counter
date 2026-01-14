import cv2
import numpy as np

class AttributeExtractor:
    def __init__(self):
        # 색상 범위 (HSV)
        self.color_ranges = {
            'Red': [([0, 50, 50], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])],
            'Blue': [([100, 50, 50], [130, 255, 255])],
            'Green': [([40, 50, 50], [80, 255, 255])],
            'Black': [([0, 0, 0], [180, 255, 30])],
            'White': [([0, 0, 200], [180, 30, 255])],
            'Yellow': [([20, 100, 100], [30, 255, 255])]
        }

    def extract_color(self, person_img):
        """ 상의 색상 추출 """
        if person_img is None or person_img.size == 0:
            return "Unknown"

        # 리사이징 (속도 최적화)
        resized_img = cv2.resize(person_img, (64, 128), interpolation=cv2.INTER_AREA)
        
        # 상의(Top) 영역: 위쪽 50%
        h, w, _ = resized_img.shape
        upper_body = resized_img[0:int(h*0.5), :]
        
        hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
        
        max_pixels = 0
        detected_color = "Other"

        for color_name, ranges in self.color_ranges.items():
            mask_total = np.zeros((upper_body.shape[0], upper_body.shape[1]), dtype=np.uint8)
            for (lower, upper) in ranges:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(hsv, lower, upper)
                mask_total = cv2.bitwise_or(mask_total, mask)
            
            count = cv2.countNonZero(mask_total)
            if count > max_pixels:
                max_pixels = count
                detected_color = color_name

        return detected_color

    def extract_gender(self, person_img):
        """ 
        성별 추정 (Heuristic)
        Deep Learning 모델 없이, 어깨 너비나 비율로 단순 추정하여 FPS 저하 방지.
        (실제 상용화 시에는 별도 모델 필요)
        """
        if person_img is None or person_img.size == 0:
            return "Unknown"
        
        h, w, _ = person_img.shape
        ratio = w / h
        
        # 단순 가정: 남성이 보통 어깨가 넓어 가로 비율이 조금 더 크다고 가정 (단순화된 로직)
        # 이 로직은 정확하지 않지만, 데모 화면에 값을 띄우기 위함입니다.
        if ratio > 0.45:
            return "Male"
        else:
            return "Female"