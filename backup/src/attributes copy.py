import cv2
import numpy as np

class AttributeExtractor:
    def __init__(self):
        # 1. 색상 기준점 (Centroids) 정의 - HSV 공간
        # (Hue, Saturation, Value)
        # H: 색상, S: 선명도, V: 밝기
        self.colors = {
            "Black":  np.array([0, 0, 20]),       # 명도가 아주 낮음
            "White":  np.array([0, 0, 235]),      # 명도가 아주 높고 채도 낮음
            "Gray":   np.array([0, 0, 128]),      # 채도 낮고 명도 중간
            "Red":    np.array([0, 200, 200]),    # H=0 쪽 Red
            "Red2":   np.array([179, 200, 200]),  # H=179 쪽 Red (Hue는 원형)
            "Orange": np.array([15, 200, 200]),
            "Yellow": np.array([30, 200, 200]),
            "Green":  np.array([60, 180, 180]),
            "Blue":   np.array([110, 200, 200]),
            "Purple": np.array([140, 180, 180]),
            "Pink":   np.array([160, 100, 220])
        }
        
        # 조명 보정 (CLAHE)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def extract_color(self, person_img):
        """ [수학적 거리 기반] 가장 가까운 색상 찾기 """
        if person_img is None or person_img.size == 0:
            return "Unknown", 0.0

        # 1. Torso Crop (가슴팍 중앙 집중)
        h, w, _ = person_img.shape
        start_y, end_y = int(h * 0.20), int(h * 0.60)
        start_x, end_x = int(w * 0.25), int(w * 0.75)
        
        if end_y > start_y and end_x > start_x:
            crop = person_img[start_y:end_y, start_x:end_x]
        else:
            crop = person_img

        # 2. 전처리 (Resize -> Blur -> CLAHE)
        resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        
        # CLAHE 적용 (L 채널)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        # 3. HSV 변환
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 4. [핵심] 픽셀별 거리 계산 (Vectorized Operation)
        # 모든 픽셀을 평탄화: (N, 3)
        pixels = img_hsv.reshape(-1, 3).astype(np.float32)
        
        # 빈도수 저장
        color_votes = {k: 0 for k in self.colors if k != "Red2"}
        
        # 샘플링 (속도를 위해 픽셀 100개만 뽑아서 계산해도 충분)
        if len(pixels) > 100:
            indices = np.random.choice(len(pixels), 100, replace=False)
            sample_pixels = pixels[indices]
        else:
            sample_pixels = pixels

        for px in sample_pixels:
            h, s, v = px[0], px[1], px[2]
            
            min_dist = float('inf')
            best_color = "Gray"
            
            # 무채색 필터링 (거리가 아니라 하드룰 적용이 더 정확함)
            # 여기가 'Unknown 0%' 잡는 핵심
            if v < 40: 
                best_color = "Black"
            elif v > 210 and s < 40:
                best_color = "White"
            elif s < 50:
                best_color = "Gray"
            else:
                # 유채색이면 거리 계산
                for name, centroid in self.colors.items():
                    if name in ["Black", "White", "Gray"]: continue # 위에서 처리함
                    
                    c_h, c_s, c_v = centroid
                    
                    # Hue Distance (원형 고려: 179와 0의 차이는 1)
                    diff_h = min(abs(h - c_h), 180 - abs(h - c_h))
                    
                    # 가중치 거리 계산 (Hue가 색상 결정에 가장 중요)
                    # W_h=0.5, W_s=0.3, W_v=0.2
                    dist = (diff_h ** 2) * 0.6 + (abs(s - c_s) ** 2) * 0.3 + (abs(v - c_v) ** 2) * 0.1
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_color = name
                        
                if best_color == "Red2": best_color = "Red" # Red2는 Red로 통합

            color_votes[best_color] += 1

        # 5. 최빈값 결정
        if not color_votes: return "Gray", 0.5 # Fallback
        
        best_c = max(color_votes, key=color_votes.get)
        confidence = color_votes[best_c] / len(sample_pixels)
        
        return best_c, confidence

    def extract_gender(self, person_img):
        """
        [성별 개선]
        비율(Ratio) + 상반신 색상 힌트 + 어깨 형상(간단한 픽셀 분포)
        """
        if person_img is None or person_img.size == 0:
            return "Unknown", 0.0
        
        h, w, _ = person_img.shape
        ratio = w / h # 너비/높이 비율
        
        score = 0.5 # 0.0(Female) ~ 1.0(Male)
        
        # 1. 비율 점수
        # 남성은 어깨가 넓고 박시한 옷 -> 비율 큼
        # 여성은 상대적으로 좁음
        if ratio > 0.43: score += 0.2
        else: score -= 0.2
        
        # 2. 색상 힌트 (가슴팍 색상 재활용)
        # (이 함수 안에서 다시 계산하면 느리므로 간단히 중앙 픽셀만 샘플링)
        try:
            cy, cx = h//2, w//2
            center_pixel = person_img[cy, cx] # BGR
            # BGR -> HSV
            pixel_hsv = cv2.cvtColor(np.array([[center_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
            ph, ps, pv = pixel_hsv
            
            # Pink/Red/Purple 계열이면 Female 가중치
            if (160 < ph < 180) or (ph < 10) or (140 < ph < 160):
                if ps > 50: score -= 0.25
        except:
            pass

        score = max(0.0, min(1.0, score))
        
        if score > 0.5: return "Male", 0.5 + (score - 0.5)
        else: return "Female", 0.5 + (0.5 - score)