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
# #2 Advanced VLM Engine (Gender & Top Confidence)
# =========================
class VLMAttributeEngine:
    def __init__(self):
        self.available = False
        self.model_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

        try:
            import clip
            import torch
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
            import clip

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root=self.model_dir)
            
            # 텍스트 쿼리 최적화
            self.genders = ["a photo of a man", "a photo of a woman"]
            self.top_styles = ["black clothing", "white clothing", "blue clothing", "red clothing", "gray clothing", "yellow clothing"]
            
            self.gender_tokens = clip.tokenize(self.genders).to(self.device)
            self.top_tokens = clip.tokenize(self.top_styles).to(self.device)
            
            self.available = True
            print(f"[INFO] VLM Engine Focused: Gender & Top Clothing.")
        except Exception as e:
            print(f"[WARN] VLM Initialization Failed: {e}")

    def analyze(self, crop_cv2):
        if not self.available: return "Unknown", "N/A", 0, 0
        from PIL import Image
        import torch
        try:
            h, w = crop_cv2.shape[:2]
            # 상의 위주 분석을 위해 상단 65% 크롭
            upper_crop = crop_cv2[0:int(h*0.65), :] 
            
            image = Image.fromarray(cv2.cvtColor(upper_crop, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits_g, _ = self.model(image_input, self.gender_tokens)
                logits_t, _ = self.model(image_input, self.top_tokens)
                
                # Softmax로 확률(Confidence) 계산
                prob_g = logits_g.softmax(dim=-1)
                prob_t = logits_t.softmax(dim=-1)
                
                g_idx = prob_g.argmax().item()
                t_idx = prob_t.argmax().item()
                
                gender = "Male" if g_idx == 0 else "Female"
                top_color = self.top_styles[t_idx].replace(" clothing", "").capitalize()
                
                # 성별 확신도와 색상 확신도 개별 추출
                g_conf = prob_g[0][g_idx].item() * 100
                t_conf = prob_t[0][t_idx].item() * 100

            return gender, top_color, g_conf, t_conf
        except:
            return "Unknown", "N/A", 0, 0

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

    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    id_attributes = {} 
    
    gender_stats = {"Male": 0, "Female": 0, "Unknown": 0}
    color_stats = defaultdict(int)
    inference_log = []

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

                if tid not in id_attributes:
                    if track_hits[raw_id] == 25:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            g, c, g_cf, t_cf = vlm_engine.analyze(crop)
                            id_attributes[tid] = (g, c, g_cf, t_cf)
                            gender_stats[g] += 1
                            color_stats[c] += 1
                            inference_log.append(f"[ID {tid:02d}] {g:6} ({g_cf:.1f}%) | Top: {c:6} ({t_cf:.1f}%)")
                
                attr = id_attributes.get(tid, ("Analyzing", "...", 0, 0))
                # 화면 표시: ID:1 | M(92%) | Blue(88%)
                label = f"ID:{tid} | {attr[0][:1]}({attr[2]:.0f}%) | {attr[1]}({attr[3]:.0f}%)"

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * src_fps):
                        counted_ids.add(tid)
                        total_count += 1

                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.8, color_box, 1)

        # Dashboard UI
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"VLM ANALYTICS COUNT: {total_count}", (20, 45), 1, 1.8, (255, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        cv2.putText(display, f"ENGINE: CLIP (Gender+Top Conf Analysis)", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("AI Tracking System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    elapsed = time.time() - start_time
    total_ids = next_display_id - 1
    
    print("\n" + "=" * 80)
    print(f"[FINAL REPORT] FOCUSED VLM ATTRIBUTE ANALYSIS (DETAILED CONFIDENCE)")
    print("=" * 80)
    print(f" 1. GENDER & COLOR DISTRIBUTION")
    print(f"    - Total Unique Tracked : {total_ids}")
    print(f"    - Gender Ratio         : Male {gender_stats['Male']} / Female {gender_stats['Female']}")
    print(f"    - Top Clothing Colors  : {dict(color_stats)}")
    print("-" * 80)
    print(f" 2. DETAILED ANALYSIS LOGS (Gender Conf | Top Conf)")
    for log in sorted(inference_log):
        print(f"    > {log}")
    print("-" * 80)
    print(f" 3. PERFORMANCE & QUALITY")
    print(f"    - Effective FPS        : {frame_idx/elapsed:.1f} (4070Ti Optimized)")
    print(f"    - Accuracy (Target 25) : {min(total_count/25, 25/total_count)*100:.1f}%")
    print("=" * 80)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()