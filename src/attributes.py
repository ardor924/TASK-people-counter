import cv2
import numpy as np

class AttributeExtractor:
    def __init__(self):
        # 1. 표준 색상 팔레트 (Lab 기준)
        self.palette = {
            "Black":  (20, 20, 20),
            "White":  (230, 230, 230),
            "Gray":   (128, 128, 128),
            "Red":    (200, 30, 30),
            "Orange": (230, 100, 30),
            "Yellow": (240, 240, 40),
            "Green":  (30, 150, 30),
            "Blue":   (20, 50, 180),     # 파랑 기준점 (더 쨍하게)
            "Navy":   (10, 10, 80),
            "Purple": (128, 0, 128),
            "Pink":   (240, 100, 180),
            "Brown":  (100, 60, 30)
        }
        
        # RGB -> Lab 변환
        self.lab_palette = {}
        for name, rgb in self.palette.items():
            pixel = np.array([[rgb]], dtype=np.uint8)
            lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)[0][0]
            self.lab_palette[name] = lab.astype(np.float32)

        # CLAHE (조명 보정)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))

    def apply_clahe(self, img):
        """ 조명 보정: 어두운 옷을 밝게 펴줌 """
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = self.clahe.apply(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def extract_attributes(self, person_img):
        if person_img is None or person_img.size == 0:
            return "Unk", 0.0, "Unk", 0.0

        h, w, _ = person_img.shape
        
        # 1. ROI 크롭 (상반신 집중)
        crop = person_img[int(h*0.15):int(h*0.55), int(w*0.20):int(w*0.80)]
        if crop.size == 0: crop = person_img

        # 2. CLAHE 전처리
        enhanced_img = self.apply_clahe(crop)

        # 3. K-Means (지배적 색상 추출)
        small_img = cv2.resize(enhanced_img, (32, 32), interpolation=cv2.INTER_AREA)
        data = small_img.reshape((-1, 3)).astype(np.float32)
        
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            dominant_bgr = centers[0].astype(np.uint8)
            dominant_rgb = dominant_bgr[::-1]
        except:
            return "Unk", 0.0, "Unk", 0.0

        # 4. Lab 거리 계산 (기본 판별)
        target_lab = cv2.cvtColor(np.array([[dominant_rgb]], dtype=np.uint8), cv2.COLOR_RGB2Lab)[0][0].astype(np.float32)
        
        min_dist = float('inf')
        detected_color = "Gray"

        for name, ref_lab in self.lab_palette.items():
            dist = np.linalg.norm(target_lab - ref_lab)
            if dist < min_dist:
                min_dist = dist
                detected_color = name
        
        # 5. [핵심] RGB Bias (회색/검정으로 죽은 파란색 살리기)
        # B 채널이 R, G보다 유의미하게 높으면 무조건 Blue로 판정
        b, g, r = dominant_bgr
        if detected_color in ["Black", "Gray", "White"]:
            if (b > r + 15) and (b > g + 10): 
                detected_color = "Blue"
            elif (r > b + 20) and (r > g + 20): # 붉은 계열
                detected_color = "Red"

        if detected_color == "Navy": detected_color = "Blue"

        color_conf = max(0.5, 1.0 - (min_dist / 130.0))

        # 6. 성별 판별
        ratio = w / h
        gender = "Male" 
        gender_conf = 0.7

        # 핑크/보라거나 비율이 좁으면 여성
        if ratio < 0.32 or detected_color in ["Pink", "Purple"]:
            gender = "Female"
            gender_conf = 0.8
        
        return gender, gender_conf, detected_color, color_conf