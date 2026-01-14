import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class AttributeExtractor:
    def __init__(self):
        # -----------------------------------------------------------
        # 1. [Color] HSV 기준점
        # -----------------------------------------------------------
        self.colors = {
            "Black":  np.array([0, 0, 20]),
            "White":  np.array([0, 0, 230]),
            "Gray":   np.array([0, 0, 128]),
            "Red":    np.array([0, 160, 180]),
            "Red2":   np.array([179, 160, 180]),
            "Orange": np.array([15, 160, 200]),
            "Yellow": np.array([30, 160, 200]),
            "Green":  np.array([60, 140, 160]),
            "Blue":   np.array([110, 150, 180]),
            "Purple": np.array([140, 140, 180]),
            "Pink":   np.array([165, 120, 200])
        }
        # 조명 보정용
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # -----------------------------------------------------------
        # 2. [Gender/Color] CLIP 모델
        # -----------------------------------------------------------
        self.model_id = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = CLIPModel.from_pretrained(self.model_id, use_safetensors=True).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
        except Exception as e:
            print(f"❌ CLIP Load Error: {e}")
            raise e
        
        # [튜닝 1] 프롬프트 단순화 (CCTV 저화질에서 인식률 상승)
        self.gender_labels = ["man", "woman"] 
        
        self.color_labels = [
            "black clothes", "white clothes", "gray clothes", 
            "red clothes", "yellow clothes", "blue clothes", 
            "green clothes", "pink clothes", "purple clothes", 
            "orange clothes", "brown clothes"
        ]

    def extract_attributes(self, person_img):
        """
        Hybrid Attribute Recognition:
        - Color: CLIP(일반화) + HSV(검증/보정)
        - Gender: CLIP(60%) + Heuristic(40%) [Ensemble]
        """
        if person_img is None or person_img.size == 0:
            return "Unknown", 0.0, "Unknown", 0.0

        h, w, _ = person_img.shape

        # --- [1] Color Analysis ---
        # 가슴팍 크롭 (배경 제거)
        start_y, end_y = int(h * 0.20), int(h * 0.60)
        start_x, end_x = int(w * 0.25), int(w * 0.75)
        
        if end_y > start_y and end_x > start_x:
            torso_crop = person_img[start_y:end_y, start_x:end_x]
        else:
            torso_crop = person_img

        # CLIP 추론 (Color)
        _, _, clip_color, clip_c_conf = self._analyze_clip_color(torso_crop)
        # HSV 검증 (Color)
        final_color, final_c_conf = self._verify_color_with_hsv(torso_crop, clip_color, clip_c_conf)


        # --- [2] Gender Analysis (Ensemble) ---
        # 성별은 '전신'을 봐야 정확함 (머리스타일, 비율, 치마 등)
        
        # A. CLIP Score (AI의 직감)
        clip_gender, clip_g_score = self._analyze_clip_gender(person_img) # Male=0.0~1.0 (Low is Male)
        
        # B. Heuristic Score (통계적 특징)
        # 0.0(Female) ~ 1.0(Male)
        heuristic_score = self._analyze_gender_heuristic(person_img, final_color)
        
        # C. 앙상블 (가중 합산)
        # CLIP 60% + Heuristic 40%
        # clip_g_score가 Male일 확률이라고 가정 (Male=Label 0)
        # CLIP 로직에서 probs[0]은 "man"일 확률
        
        final_male_prob = (clip_g_score * 0.6) + (heuristic_score * 0.4)
        
        if final_male_prob > 0.5:
            final_gender = "Male"
            final_g_conf = final_male_prob
        else:
            final_gender = "Female"
            final_g_conf = 1.0 - final_male_prob

        return final_gender, final_g_conf, final_color, final_c_conf

    def _analyze_clip_gender(self, img):
        """ CLIP: Returns ('Male'/'Female', male_probability) """
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            inputs = self.processor(
                text=self.gender_labels, images=pil_img, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            # probs[0][0] = 'man' 확률, probs[0][1] = 'woman' 확률
            male_prob = probs[0][0].item()
            
            gender = "Male" if male_prob > 0.5 else "Female"
            return gender, male_prob
        except:
            return "Male", 0.5 # Fallback

    def _analyze_clip_color(self, img):
        """ CLIP: Color only """
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            inputs = self.processor(
                text=self.color_labels, images=pil_img, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                probs = self.model(**inputs).logits_per_image.softmax(dim=1)
            
            idx = probs.argmax().item()
            conf = probs[0][idx].item()
            
            raw = self.color_labels[idx]
            color_str = raw.replace(" clothes", "").capitalize()
            return "Unknown", 0.0, color_str, conf
        except:
            return "Unknown", 0.0, "Gray", 0.0

    def _analyze_gender_heuristic(self, img, detected_color):
        """
        [규칙 기반 성별 점수]
        Returns: 0.0 (Female) ~ 1.0 (Male)
        """
        h, w, _ = img.shape
        ratio = w / h
        
        score = 0.5 # 중립 시작
        
        # 1. 비율 점수 (어깨가 넓으면 남성 가산점)
        # 서 있는 사람 기준 w/h는 보통 0.3 ~ 0.5
        if ratio > 0.45: score += 0.25 # 뚱뚱하거나 어깨 넓음 -> 남성 확률 Up
        elif ratio < 0.35: score -= 0.15 # 매우 슬림 -> 여성 확률 Up (약하게)

        # 2. 색상 바이어스 (강력한 힌트)
        female_colors = ["Pink", "Purple", "Red", "Orange", "Yellow"]
        male_colors = ["Blue", "Black", "Gray", "Green"]
        
        if detected_color in female_colors:
            score -= 0.35 # 여성일 확률 대폭 증가
        elif detected_color in male_colors:
            score += 0.1  # 남성일 확률 소폭 증가 (남색 옷 입은 여자도 많으므로 약하게)

        # 점수 클리핑 (0.0 ~ 1.0)
        return max(0.0, min(1.0, score))

    def _verify_color_with_hsv(self, img, clip_color, clip_conf):
        """ HSV 검증 (기존 로직 유지) """
        resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        s_median = np.median(hsv[:, :, 1])
        v_median = np.median(hsv[:, :, 2])
        
        if v_median < 45: return "Black", 0.95
        if s_median < 70:
            if v_median > 180: return "White", 0.90
            else: return "Gray", 0.90
            
        return clip_color, clip_conf

    # 호환성
    def extract_color(self, img): pass
    def extract_gender(self, img): pass