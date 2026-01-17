import os
import sys
import subprocess
import numpy as np
import cv2
from PIL import Image

# [중요] 모듈 로딩 에러 방지를 위해 전역에서 import
try:
    import torch
    import clip
except ImportError:
    print("[INFO] Libraries missing. Installing CLIP & Torch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    import torch
    import clip

class VLMAttributeEngine:
    def __init__(self):
        self.available = False
        self.model_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"[INFO] Loading CLIP model to {self.device.upper()}...")
            self.model, self.preprocess = clip.load("ViT-B/16", device=self.device, download_root=self.model_dir)
            
            # [1] 성별 프롬프트
            self.m_prompts = ["a photo of a man", "a male person", "this is a man", "a gentleman", "a guy"]
            self.w_prompts = ["a photo of a woman", "a female person", "this is a woman", "a lady", "a girl"]
            self.all_g_prompts = self.m_prompts + self.w_prompts
            self.m_len = len(self.m_prompts)
            
            # [2] 색상 프롬프트 (상의 집중 강화)
            self.color_names = [
                "Black", "White", "Grey", "Red", "Blue", "Yellow", "Green", "Orange"
            ]
            
            # [수정] 'fabric shirt' -> 'upper body clothing'으로 변경하여 하의 배제 의도 전달
            self.color_prompts = [
                f"a close-up photo of a {c.lower()} color upper body clothing, shirt or jacket" for c in self.color_names
            ]
            
            # 토큰화
            self.gender_tokens = clip.tokenize(self.all_g_prompts).to(self.device)
            self.color_tokens = clip.tokenize(self.color_prompts).to(self.device)
            
            self.available = True
            print(f"[INFO] VLM Engine: Upper-Chest Focus Mode Active.")
        except Exception as e:
            print(f"[WARN] VLM Init Failed: {e}")

    def analyze(self, crop_cv2):
        if not self.available: return None
        
        try:
            h, w = crop_cv2.shape[:2]
            
            # [안전장치] 이미지가 너무 작으면 분석 불가
            if h < 10 or w < 10: return None

            # [전략 1] 성별용 크롭 (다리 제외, 상체 위주)
            crop_g_cv = crop_cv2[0:int(h*0.7), int(w*0.05):int(w*0.95)]
            if crop_g_cv.size == 0: crop_g_cv = crop_cv2
            
            img_g = Image.fromarray(cv2.cvtColor(crop_g_cv, cv2.COLOR_BGR2RGB))
            input_g = self.preprocess(img_g).unsqueeze(0).to(self.device)

            # [전략 2] 색상용 크롭 (Upper Chest Focus)
            # 바지 간섭을 피하기 위해 높이를 h*0.38 (가슴 중앙)까지만 제한
            # ROI: 목 아래(12%) ~ 명치 위(38%) -> 오직 상의만 존재하는 구간
            roi_y1, roi_y2 = int(h*0.12), int(h*0.38)
            roi_x1, roi_x2 = int(w*0.30), int(w*0.70)
            
            # 너무 작아서 크롭이 안되면 상반신 전체 사용
            if roi_y2 - roi_y1 < 5 or roi_x2 - roi_x1 < 5:
                crop_c_cv = crop_cv2[0:int(h*0.5), :]
            else:
                crop_c_cv = crop_cv2[roi_y1:roi_y2, roi_x1:roi_x2]

            if crop_c_cv.size == 0: crop_c_cv = crop_cv2 

            img_c = Image.fromarray(cv2.cvtColor(crop_c_cv, cv2.COLOR_BGR2RGB))
            input_c = self.preprocess(img_c).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # --- A. 성별 추론 ---
                logits_g, _ = self.model(input_g, self.gender_tokens)
                probs_g = (logits_g / 0.7).softmax(dim=-1).cpu().numpy()[0]
                m_score = np.sum(probs_g[:self.m_len])
                w_score = np.sum(probs_g[self.m_len:])
                best_g_text = self.all_g_prompts[np.argmax(probs_g)]

                # --- B. 색상 추론 ---
                logits_c, _ = self.model(input_c, self.color_tokens)
                probs_c = (logits_c / 0.6).softmax(dim=-1).cpu().numpy()[0]
                c_idx = np.argmax(probs_c)
                
                vlm_color = self.color_names[c_idx]
                vlm_conf = probs_c[c_idx] * 100
                best_c_text = self.color_prompts[c_idx]

            # --- C. HSV 검증 (단순화된 버전) ---
            final_color = vlm_color
            final_c_conf = vlm_conf
            correction_note = ""

            # 작은 블러 처리
            if crop_c_cv.shape[0] > 5 and crop_c_cv.shape[1] > 5:
                blurred_roi = cv2.GaussianBlur(crop_c_cv, (5, 5), 0)
            else:
                blurred_roi = crop_c_cv
                
            hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
            mean_s = np.mean(hsv_roi[:, :, 1]) # Saturation
            mean_v = np.mean(hsv_roi[:, :, 2]) # Value

            # [규칙 1] 채도가 매우 낮으면(30 미만) 무조건 무채색
            if mean_s < 30:
                if mean_v < 50: final_color = "Black"
                elif mean_v > 200: final_color = "White"
                else: final_color = "Grey"
                if final_color != vlm_color: correction_note = " (HSV Forced)"
            
            # [규칙 2] Black/White 명도 검증
            if final_color == "Black" and mean_v > 110:
                final_color = "Grey"
            if final_color == "White" and mean_v < 150:
                final_color = "Grey"

            gender = "Male" if m_score > w_score else "Female"
            total_g = m_score + w_score
            g_conf = (max(m_score, w_score) / total_g) * 100 if total_g > 0 else 0
            
            description = f"Matched: '{best_g_text}' & '{best_c_text}'{correction_note}"

            return {"g": gender, "c": final_color, "g_cf": g_conf, "c_cf": final_c_conf, "desc": description}
            
        except Exception as e:
            print(f"[VLM ERROR] {e}") 
            return None