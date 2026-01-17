import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict, Counter
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import create_directories, save_best_samples, generate_report, show_summary_window

# =========================
# #1 Global Policy (Low-Spec)
# =========================
# VIDEO_PATH = "data/sample.avi"
VIDEO_PATH = "data/dev_day.mp4"
# VIDEO_PATH = "data/eval_day.mp4"
# VIDEO_PATH = "data/eval_night.mp4"
# VIDEO_PATH = "data/eval_indoors.mp4"
DETECTOR_PATH = "models/yolov8n.pt"
CLASSIFIER_PATH = "models/yolov8n-cls.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.65       
# CONF_THRESH = 0.62       
# DWELL_TIME_SEC = 0.5     
DWELL_TIME_SEC = 0.48     
FORGET_TIME_SEC = 0.7    
TRACK_BUFFER_SEC = 2.0   
ZONE_START_RATIO = 0.45

# =========================
# #2 Hybrid Engine
# =========================
class HybridAttributeEngine:
    def __init__(self, device="cuda"):
        from ultralytics import YOLO
        self.device = device
        
        try:
            self.classifier = YOLO(CLASSIFIER_PATH).to(self.device)
            print(f"[INFO] Gender Classifier Loaded on {self.device}.")
        except Exception as e:
            print(f"[WARN] Failed to load classifier: {e}")
            self.classifier = None

        self.color_centers = {
            "Black":  [0, 0, 30], "White":  [0, 0, 220], "Red":    [0, 200, 150],
            "Blue":   [115, 200, 150], "Yellow": [25, 200, 200], "Gray":   [0, 0, 100],
            "Green":  [60, 150, 100]
        }

    def analyze(self, crop_cv2):
        res_g, g_conf = "Unk", 0.0
        res_c, c_conf = "Unk", 0.0

        if crop_cv2.size == 0:
            return {"g": res_g, "c": res_c, "g_cf": 0, "c_cf": 0}

        if self.classifier:
            try:
                res = self.classifier(crop_cv2, verbose=False, device=self.device)[0]
                top1_idx = res.probs.top1
                conf = res.probs.top1conf.item() * 100
                class_name = res.names[top1_idx].lower()
                if 'woman' in class_name or 'female' in class_name: res_g = 'Female'
                elif 'man' in class_name or 'male' in class_name: res_g = 'Male'
                else: res_g = "Male" if top1_idx % 2 != 0 else "Female"
                g_conf = conf
            except: pass

        try:
            h, w = crop_cv2.shape[:2]
            roi = crop_cv2[int(h*0.2):int(h*0.5), int(w*0.3):int(w*0.7)]
            if roi.size > 0:
                pixels = roi.reshape((-1, 3)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 3 if len(pixels) >= 3 else 1
                _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                counts = np.bincount(labels.flatten())
                total = len(roi.reshape((-1, 3)))
                max_score = -1.0

                for i, center in enumerate(centers):
                    weight = counts[i] / total
                    if weight < 0.15: continue
                    c_bgr = np.uint8([[center]])
                    c_hsv = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2HSV)[0][0]
                    is_skin = (5 <= c_hsv[0] <= 25) and (40 <= c_hsv[1] <= 180)
                    min_dist = 9999
                    temp_c = "Unk"
                    for name, target in self.color_centers.items():
                        h_diff = abs(float(c_hsv[0]) - target[0])
                        if h_diff > 90: h_diff = 180 - h_diff
                        dist = np.sqrt(2.5*(h_diff**2) + 1.0*((float(c_hsv[1])-target[1])**2) + 0.8*((float(c_hsv[2])-target[2])**2))
                        if dist < min_dist:
                            min_dist = dist
                            temp_c = name
                    score = (1000 - min_dist) * weight
                    if is_skin: score *= 0.3
                    if score > max_score:
                        max_score = score
                        res_c = temp_c
                        c_conf = max(0, min(100, (1 - min_dist/300) * 100))
        except: pass
        return {"g": res_g, "c": res_c, "g_cf": g_conf, "c_cf": c_conf}

# =========================
# #3 Main Pipeline
# =========================
def main():
    import supervision as sv
    from ultralytics import YOLO

    create_directories()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running in Hybrid Voting Mode (Device: {device})")

    detector = YOLO(DETECTOR_PATH).to(device)
    hybrid_engine = HybridAttributeEngine(device=device)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream.")
        return
    # [FPS Control] Smart Sync + 30 FPS Limit
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = min(raw_fps, 30.0) if raw_fps > 0 else 30.0
    frame_interval_ms = int(1000 / target_fps)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Target Video: {os.path.basename(VIDEO_PATH)}")
    print(f"[INFO] Speed Control: Native {raw_fps:.1f} FPS -> Sync to {target_fps:.1f} FPS")
    
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * target_fps))

    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    voting_box = defaultdict(lambda: {'g_scores': defaultdict(float), 'c_list': [], 'best_img': None, 'max_conf': 0})
    id_attributes = {} 
    best_crops_store = {}   
    inference_log = []      
    
    # [수정] 신뢰도 분리
    conf_scores_g = []        
    conf_scores_c = []        
    
    gender_stats = {"Male": 0, "Female": 0, "Unk": 0}
    frame_idx = 0
    start_time = time.time()
    
    cv2.namedWindow("Hybrid Low-Spec AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid Low-Spec AI", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    print(f"[INFO] Analysis Started...")

    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        fh, fw = int(frame.shape[0] * (RESIZE_WIDTH/frame.shape[1])), RESIZE_WIDTH
        frame = cv2.resize(frame, (fw, fh))
        display = frame.copy()
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        results = detector(frame, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        valid_indices = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            aspect_ratio = bh / bw if bw > 0 else 0
            if aspect_ratio > 1.15 and detections.confidence[i] >= CONF_THRESH:
                valid_indices.append(i)
        
        detections = detections[valid_indices]
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()

        if detections.tracker_id is not None:
            for xyxy, raw_id, conf in zip(detections.xyxy, detections.tracker_id, detections.confidence):
                raw_id = int(raw_id)
                track_hits[raw_id] += 1
                if track_hits[raw_id] < 3: continue 

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if raw_id not in id_map:
                    found_legacy = False
                    for disp_id, data in list(track_registry.items()):
                        if disp_id in active_now_ids: continue 
                        if frame_idx - data["last_frame"] > (FORGET_TIME_SEC * target_fps):
                            if disp_id in track_registry: del track_registry[disp_id]
                            continue
                        dist = np.sqrt((cx - data["pos"][0])**2 + (cy - data["pos"][1])**2)
                        if dist < (RESIZE_WIDTH * 0.02):
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
                    if track_hits[raw_id] in [3, 10, 20]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = hybrid_engine.analyze(crop)
                            if res['g'] in ['Male', 'Female']:
                                voting_box[tid]['g_scores'][res['g']] += res['g_cf']
                            if res['c'] != "Unk":
                                voting_box[tid]['c_list'].append(res['c'])
                            curr_score = res['g_cf'] + res['c_cf']
                            if curr_score > voting_box[tid]['max_conf']:
                                voting_box[tid]['max_conf'] = curr_score
                                voting_box[tid]['best_img'] = crop.copy()
                                
                    # 31프레임째에 확정 (기존 로직)
                    if track_hits[raw_id] == 31:
                        vdata = voting_box[tid]
                        m_score = vdata['g_scores']['Male']
                        f_score = vdata['g_scores']['Female']
                        
                        final_g = "Unk"
                        if (m_score + f_score) > 50: 
                            final_g = "Male" if m_score >= f_score else "Female"
                        
                        final_c = "Unk"
                        if vdata['c_list']:
                            final_c = Counter(vdata['c_list']).most_common(1)[0][0]
                        
                        id_attributes[tid] = {"g": final_g, "c": final_c}
                        
                        if final_g in gender_stats:
                            gender_stats[final_g] += 1
                        else:
                            gender_stats['Unk'] += 1
                        
                        # 신뢰도 기록
                        log_g_conf = max(m_score, f_score) / 3
                        log_c_conf = (vdata['max_conf'] - log_g_conf) if vdata['max_conf'] > log_g_conf else 0.0
                        
                        conf_scores_g.append(log_g_conf)
                        conf_scores_c.append(log_c_conf) 
                        
                        inference_log.append(f"[ID {tid:02d}] Final: {final_g}({log_g_conf:.1f}%) | {final_c}")
                        
                        if vdata['best_img'] is not None:
                            best_crops_store[tid] = {
                                'conf': log_g_conf,
                                'img': vdata['best_img'],
                                'label': f"{final_g}_{final_c}"
                            }

                attr = id_attributes.get(tid, {"g": "Scaning..", "c": "."})
                label = f"ID:{tid} | {attr['g'][0]}/{attr['c']}"

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * target_fps):
                        counted_ids.add(tid)
                        total_count += 1

                color_box = (0, 255, 0) if tid in id_attributes else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.7, color_box, 1)

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"HYBRID VOTING SYSTEM: {os.path.basename(VIDEO_PATH)}", (20, 45), 1, 1.4, (0, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        
        real_fps = frame_idx / (time.time() - start_time)
        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(display, f"PROG : {int(progress)}% | FPS : {real_fps:.1f} (Limit: {target_fps:.0f})", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("Hybrid Low-Spec AI", display)
        
        proc_time_ms = (time.time() - loop_start_time) * 1000
        wait_ms = max(1, int(frame_interval_ms - proc_time_ms))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    final_fps = frame_idx / elapsed
    
    # [수정] 평균 계산
    avg_g_conf = np.mean(conf_scores_g) if conf_scores_g else 0.0
    avg_c_conf = np.mean(conf_scores_c) if conf_scores_c else 0.0

    video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    save_best_samples(video_base_name, best_crops_store)
    
    # [수정] 인자 전달
    generate_report(video_base_name, total_count, gender_stats, avg_g_conf, avg_c_conf, final_fps, inference_log, file_prefix="LOW_")
    show_summary_window(video_base_name, total_count, gender_stats, avg_g_conf, avg_c_conf, final_fps)

if __name__ == "__main__":
    main()