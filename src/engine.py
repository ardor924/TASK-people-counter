# src/engine.py
import os
import sys
import subprocess
import numpy as np
import cv2

class VLMAttributeEngine:
    def __init__(self):
        self.available = False
        self.model_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

        # CLIP 라이브러리 확인 및 설치
        try:
            import clip
            import torch
        except ImportError:
            print("[INFO] CLIP library missing. Attempting auto-installation...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
            import clip

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # ViT-B/16 모델 로드
            print(f"[INFO] Loading CLIP model to {self.device.upper()}...")
            self.model, self.preprocess = clip.load("ViT-B/16", device=self.device, download_root=self.model_dir)
            
            # 프롬프트 정의
            self.m_prompts = ["a man with short hair", "a male person"]
            self.w_prompts = ["a woman with long hair", "a female person"]
            self.color_names = ["Black", "White", "Blue", "Red", "Gray", "Yellow"]
            self.color_prompts = [f"a person wearing {c.lower()} clothes" for c in self.color_names]
            
            # 토큰화 (GPU 업로드)
            self.gender_tokens = clip.tokenize(self.m_prompts + self.w_prompts).to(self.device)
            self.color_tokens = clip.tokenize(self.color_prompts).to(self.device)
            
            self.available = True
            print(f"[INFO] VLM Engine: Best-Frame Selection Mode (ViT-B/16) Active.")
        except Exception as e:
            print(f"[WARN] VLM Init Failed: {e}")

    def analyze(self, crop_cv2):
        if not self.available: return None
        from PIL import Image
        import torch
        
        try:
            h, w = crop_cv2.shape[:2]
            # 상단 70% ROI 추출 (바닥 노이즈 차단)
            center_crop = crop_cv2[int(h*0.05):int(h*0.75), int(w*0.1):int(w*0.9)]
            image = Image.fromarray(cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 성별 추론
                logits_g, _ = self.model(image_input, self.gender_tokens)
                probs_g = logits_g.softmax(dim=-1).cpu().numpy()[0]
                m_score = np.sum(probs_g[0:2])
                w_score = np.sum(probs_g[2:4])
                
                # 색상 추론
                logits_c, _ = self.model(image_input, self.color_tokens)
                probs_c = logits_c.softmax(dim=-1).cpu().numpy()[0]
                c_idx = np.argmax(probs_c)

            gender = "Male" if m_score > w_score else "Female"
            g_conf = max(m_score, w_score) * 100
            color = self.color_names[c_idx]
            c_conf = probs_c[c_idx] * 100

            return {"g": gender, "c": color, "g_cf": g_conf, "c_cf": c_conf}
        except:
            return None