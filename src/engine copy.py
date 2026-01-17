import os
import sys
import subprocess
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. Library Dependency Check (Fail-safe)
# ==========================================
try:
    import torch
    import clip
except ImportError:
    print("[INFO] Libraries missing. Attempting auto-installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    import torch
    import clip

class VLMAttributeEngine:
    def __init__(self):
        self.available = False
        self.model_dir = os.path.join(os.getcwd(), "models")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            print(f"[INFO] Initializing CLIP (ViT-B/16) on {self.device.upper()}...")
            self.model, self.preprocess = clip.load("ViT-B/16", device=self.device, download_root=self.model_dir)
            
            # [Attribute Definition] Prompts for Zero-shot Inference
            self.m_prompts = ["a photo of a man", "a male person", "this is a man"]
            self.w_prompts = ["a photo of a woman", "a female person", "this is a woman"]
            self.all_g_prompts = self.m_prompts + self.w_prompts
            self.m_len = len(self.m_prompts)
            
            self.color_names = ["Black", "White", "Grey", "Red", "Blue", "Yellow", "Green", "Orange"]
            self.color_prompts = [f"a close-up photo of a {c.lower()} color upper body clothing" for c in self.color_names]
            
            # Tokenization
            self.gender_tokens = clip.tokenize(self.all_g_prompts).to(self.device)
            self.color_tokens = clip.tokenize(self.color_prompts).to(self.device)
            
            self.available = True
            print(f"[INFO] VLM Engine: Hybrid HSV-Truth Filter Active.")
        except Exception as e:
            print(f"[WARN] VLM Init Failed: {e}")

    def analyze(self, crop_cv2):
        if not self.available or crop_cv2.size == 0: return None
        
        try:
            h, w = crop_cv2.shape[:2]
            if h < 10 or w < 10: return None

            # ==========================================
            # 2. Strategic Cropping (Upper-Chest Focus)
            # ==========================================
            # Gender: Focus on Upper Body
            crop_g = crop_cv2[0:int(h*0.7), int(w*0.05):int(w*0.95)]
            input_g = self.preprocess(Image.fromarray(cv2.cvtColor(crop_g, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)

            # Color: Upper-Chest Focus (12%~38% Height) to avoid pants/background
            roi_y1, roi_y2 = int(h*0.12), int(h*0.38)
            roi_x1, roi_x2 = int(w*0.30), int(w*0.70)
            crop_c = crop_cv2[roi_y1:roi_y2, roi_x1:roi_x2] if (roi_y2-roi_y1)>5 else crop_cv2[0:int(h*0.5), :]
            input_c = self.preprocess(Image.fromarray(cv2.cvtColor(crop_c, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # CLIP Inference
                logits_g, _ = self.model(input_g, self.gender_tokens)
                probs_g = (logits_g / 0.7).softmax(dim=-1).cpu().numpy()[0]
                m_score, w_score = np.sum(probs_g[:self.m_len]), np.sum(probs_g[self.m_len:])

                logits_c, _ = self.model(input_c, self.color_tokens)
                probs_c = (logits_c / 0.6).softmax(dim=-1).cpu().numpy()[0]
                c_idx = np.argmax(probs_c)
                vlm_color, vlm_conf = self.color_names[c_idx], probs_c[c_idx] * 100

            # ==========================================
            # 3. HSV Hybrid Verification (Truth Filter)
            # ==========================================
            final_color, correction_note = vlm_color, ""
            hsv_roi = cv2.cvtColor(cv2.GaussianBlur(crop_c, (5, 5), 0), cv2.COLOR_BGR2HSV)
            mean_s, mean_v = np.mean(hsv_roi[:, :, 1]), np.mean(hsv_roi[:, :, 2])

            # Force Achromatic (Black/White/Grey) if Saturation is low
            if mean_s < 30:
                if mean_v < 50: final_color = "Black"
                elif mean_v > 200: final_color = "White"
                else: final_color = "Grey"
                if final_color != vlm_color: correction_note = " (HSV Forced)"
            
            # Brightness Re-validation
            if final_color == "Black" and mean_v > 110: final_color = "Grey"
            if final_color == "White" and mean_v < 150: final_color = "Grey"

            res_g = "Male" if m_score > w_score else "Female"
            g_conf = (max(m_score, w_score) / (m_score + w_score)) * 100

            return {"g": res_g, "c": final_color, "g_cf": g_conf, "c_cf": vlm_conf, "desc": correction_note}
            
        except Exception as e:
            print(f"[VLM ERROR] {e}"); return None