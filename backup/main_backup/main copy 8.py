import cv2
import time
import numpy as np
import os
import sys
import subprocess
from collections import defaultdict

# =========================
# #1 Global Policy
# =========================
VIDEO_PATH = "data/dev_day.mp4"
MODEL_PATH = "models/yolov8n.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.62      
DWELL_TIME_SEC = 0.5    
FORGET_TIME_SEC = 0.7   
TRACK_BUFFER_SEC = 2.0  
ID_FUSION_RATIO = 0.015 
ZONE_START_RATIO = 0.45

# =========================
# #2 VLM Attribute Engine (CLIP Optimized)
# =========================
class VLMAttributeEngine:
    def __init__(self):
        self.available = False
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
            # 모델 로드 (캐시 경로: ~/.cache/clip)
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # [개선] 더 명확하고 대조적인 텍스트 쿼리 (성별 + 상의 색상 동시 추론)
            self.gender_queries = ["a photo of a man", "a photo of a woman"]
            self.color_queries = ["black shirt", "white shirt", "blue shirt", "red shirt", "gray shirt"]
            
            self.gender_tokens = clip.tokenize(self.gender_queries).to(self.device)
            self.color_tokens = clip.tokenize(self.color_queries).to(self.device)
            self.available = True
            print(f"[INFO] VLM Engine Initialized on {self.device}")
        except Exception as e:
            print(f"[WARN] VLM Initialization Failed: {e}")

    def analyze(self, crop_cv2):
        if not self.available: return "Unknown", "Unknown"
        from PIL import Image
        import torch
        try:
            image = Image.fromarray(cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 성별과 색상을 각각 독립적으로 계산하여 정확도 향상
                logits_g, _ = self.model(image_input, self.gender_tokens)
                logits_c, _ = self.model(image_input, self.color_tokens)
                
                gender_idx = logits_g.argmax().item()
                color_idx = logits_c.argmax().item()

            gender = "Male" if gender_idx == 0 else "Female"
            color = self.color_queries[color_idx].split()[0].capitalize()
            return gender, color
        except:
            return "Unknown", "Unknown"

# =========================
# #3 Main Pipeline
# =========================
def main():
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    import supervision as sv
    from ultralytics import YOLO

    detector = YOLO(MODEL_PATH)
    vlm_engine = VLMAttributeEngine()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wait_time = max(1, int(1000 / src_fps))
    
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * src_fps))

    # State
    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    id_attributes = {} 
    gender_stats = {"Male": 0, "Female": 0, "Unknown": 0}

    frame_idx, start_time = 0, time.time()
    fusion_dist_limit = RESIZE_WIDTH * ID_FUSION_RATIO

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        fh, fw = int(frame.shape[0] * (RESIZE_WIDTH/frame.shape[1])), RESIZE_WIDTH
        frame = cv2.resize(frame, (fw, fh))
        display = frame.copy()

        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        results = detector(frame, verbose=False, device=0)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence >= CONF_THRESH]
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()

        if detections.tracker_id is not None:
            for xyxy, raw_id in zip(detections.xyxy, detections.tracker_id):
                raw_id = int(raw_id)
                track_hits[raw_id] += 1
                if track_hits[raw_id] < 10: continue 

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ID Fusion
                if raw_id not in id_map:
                    found_legacy = False
                    for disp_id, data in list(track_registry.items()):
                        if disp_id in active_now_ids: continue 
                        if frame_idx - data["last_frame"] > (FORGET_TIME_SEC * src_fps):
                            if disp_id in track_registry: del track_registry[disp_id]
                            continue
                        dist = np.sqrt((cx - data["pos"][0])**2 + (cy - data["pos"][1])**2)
                        if dist < fusion_dist_limit:
                            id_map[raw_id] = disp_id
                            found_legacy = True
                            break
                    if not found_legacy:
                        id_map[raw_id] = next_display_id
                        next_display_id += 1
                
                tid = id_map[raw_id]
                active_now_ids.add(tid)
                track_registry[tid] = {"pos": (cx, cy), "last_frame": frame_idx}

                # VLM Inference (Caching)
                if tid not in id_attributes:
                    # 사람이 가장 잘 보이는 25프레임째에 분석
                    if track_hits[raw_id] == 25:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            g, c = vlm_engine.analyze(crop)
                            id_attributes[tid] = (g, c)
                            gender_stats[g] += 1
                
                attr = id_attributes.get(tid, ("Analyzing..", ""))
                label = f"ID {tid}: {attr[0]} ({attr[1]})"

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * src_fps):
                        counted_ids.add(tid)
                        total_count += 1

                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.9, color_box, 2)

        # Dashboard UI
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"VLM COUNT : {total_count}", (20, 45), 1, 1.8, (255, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        cv2.putText(display, f"SOTA ENGINE: CLIP ViT-B/32", (20, 130), 1, 1, (150, 150, 150), 1)

        cv2.imshow("AI Tracking System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    elapsed = time.time() - start_time
    total_ids = next_display_id - 1
    
    print("\n" + "=" * 60)
    print(f"[REPORT] SOTA VLM PEOPLE ANALYSIS SYSTEM")
    print("=" * 60)
    print(f" 1. GENDER DISTRIBUTION (Zero-shot)")
    print(f"    - Total Unique Persons : {total_ids}")
    print(f"    - Male Identified      : {gender_stats['Male']}")
    print(f"    - Female Identified    : {gender_stats['Female']}")
    print("-" * 60)
    print(f" 2. TRACKING & COUNTING")
    print(f"    - Final Count          : {total_count}")
    print(f"    - Accuracy (Target 25) : {min(total_count/25, 25/total_count)*100:.1f}%")
    print("-" * 60)
    print(f" 3. PERFORMANCE")
    print(f"    - Effective FPS        : {frame_idx/elapsed:.1f} (4070Ti Optimized)")
    print(f"    - Model Path           : ~/.cache/clip (Auto-managed)")
    print("=" * 60)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()