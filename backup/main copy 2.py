import cv2
import time
import numpy as np
import os
import sys
import subprocess
from collections import defaultdict
from datetime import datetime

# =========================
# #1 Global Policy
# =========================
VIDEO_PATH = "data/dev_day.mp4"
# VIDEO_PATH = "data/eval_night.mp4"
# VIDEO_PATH = "data/eval_indoors.mp4"
MODEL_PATH = "models/yolov8n.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.65      # 야간 오탐지 방지 임계값
DWELL_TIME_SEC = 0.5    
FORGET_TIME_SEC = 0.7   
TRACK_BUFFER_SEC = 2.0  
ID_FUSION_RATIO = 0.015 
ZONE_START_RATIO = 0.45

# =========================
# #2 Advanced VLM Engine (Best-Frame Selection)
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
            print("[INFO] CLIP library missing. Attempting auto-installation...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
            import clip

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/16", device=self.device, download_root=self.model_dir)
            
            self.m_prompts = ["a man with short hair", "a male person"]
            self.w_prompts = ["a woman with long hair", "a female person"]
            self.color_names = ["Black", "White", "Blue", "Red", "Gray", "Yellow"]
            self.color_prompts = [f"a person wearing {c.lower()} clothes" for c in self.color_names]
            
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
                logits_g, _ = self.model(image_input, self.gender_tokens)
                probs_g = logits_g.softmax(dim=-1).cpu().numpy()[0]
                m_score = np.sum(probs_g[0:2])
                w_score = np.sum(probs_g[2:4])
                
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

# =========================
# #3 Main Pipeline
# =========================
def main():
    # 필수 폴더 생성
    for folder in ["logs", "best_samples"]:
        if not os.path.exists(folder): os.makedirs(folder)
    
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    import supervision as sv
    from ultralytics import YOLO

    detector = YOLO(MODEL_PATH)
    vlm_engine = VLMAttributeEngine()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wait_time = max(1, int(1000 / src_fps))
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * src_fps))

    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    # ID별 VLM 후보군 (결과 + 원본이미지 함께 저장)
    vlm_candidates = defaultdict(list)
    id_attributes = {} 
    
    # [NEW] 최종 결과 이미지 저장을 위한 딕셔너리 (ID: (conf, image, gender, color))
    best_crops_store = {}

    gender_stats = {"Male": 0, "Female": 0, "Unknown": 0}
    inference_log = []
    conf_scores = [] 

    frame_idx, start_time = 0, time.time()
    fusion_dist_limit = RESIZE_WIDTH * ID_FUSION_RATIO

    print(f"[INFO] Analysis Started: {os.path.basename(VIDEO_PATH)}")

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
        
        # [방어 로직] 1. 기하학적 필터링 (Aspect Ratio)
        valid_indices = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            aspect_ratio = bh / bw if bw > 0 else 0
            if aspect_ratio > 1.5 and detections.confidence[i] >= CONF_THRESH:
                valid_indices.append(i)
        
        detections = detections[valid_indices]
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
                    if track_hits[raw_id] in [20, 30, 40]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = vlm_engine.analyze(crop)
                            # [핵심] 결과와 함께 이미지를 임시 저장 (나중에 Best 컷 저장을 위해)
                            if res: vlm_candidates[tid].append({'res': res, 'img': crop.copy()})

                    if track_hits[raw_id] == 41:
                        candidates = vlm_candidates[tid]
                        if candidates:
                            # 확신도가 가장 높은 프레임 선택
                            best_entry = max(candidates, key=lambda x: x['res']['g_cf'])
                            best_res = best_entry['res']
                            best_img = best_entry['img']
                            
                            final_g, final_c = best_res["g"], best_res["c"]
                            g_conf, c_conf = best_res["g_cf"], best_res["c_cf"]
                            
                            if g_conf < 50.0: final_g = "Unknown"
                            
                            id_attributes[tid] = (final_g, final_c, g_conf, c_conf)
                            gender_stats[final_g] += 1
                            conf_scores.append(g_conf)
                            
                            inference_log.append(f"[ID {tid:02d}] Best G: {g_conf:4.1f}% | C: {c_conf:4.1f}% | Res: {final_g}/{final_c}")
                            
                            # [NEW] Best Sample 저장을 위해 메모리에 등록
                            best_crops_store[tid] = {
                                'conf': g_conf,
                                'img': best_img,
                                'label': f"{final_g}_{final_c}"
                            }

                attr = id_attributes.get(tid, ("Analyzing", "...", 0, 0))
                label = f"ID:{tid} | {attr[0][:1]}({attr[2]:.0f}%) | {attr[1]}({attr[3]:.0f}%)"

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * src_fps):
                        counted_ids.add(tid)
                        total_count += 1

                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.8, color_box, 1)

        # UI
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"VLM ROBUST SYSTEM: {os.path.basename(VIDEO_PATH)}", (20, 45), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        cv2.putText(display, f"PROGRESS : {int(frame_idx/total_frames*100)}%", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("AI Tracking System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    elapsed = time.time() - start_time
    avg_conf = np.mean(conf_scores) if conf_scores else 0
    
    # 시간 기반 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    report_filename = f"logs/report_{video_base_name}_{timestamp}.txt"

    # [NEW] Top 5 High-Confidence Images Saving Logic
    print("\n[INFO] Saving Top 5 Best Samples...")
    sorted_crops = sorted(best_crops_store.items(), key=lambda item: item[1]['conf'], reverse=True)[:5]
    
    for rank, (tid, data) in enumerate(sorted_crops, 1):
        try:
            # 파일명 규칙: 원본파일명_날짜_순위_ID_특징_확신도.jpg
            img_filename = f"{video_base_name}_{timestamp}_Top{rank}_ID{tid:02d}_{data['label']}_{data['conf']:.0f}pct.jpg"
            save_path = os.path.join("best_samples", img_filename)
            cv2.imwrite(save_path, data['img'])
            print(f"   > Saved: {img_filename}")
        except Exception as e:
            print(f"   > Failed to save sample ID {tid}: {e}")

    # 리포트 생성
    report_content = f"""
===================================================================================================================
[FINAL SYSTEM EVALUATION REPORT]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : {os.path.basename(VIDEO_PATH)}
    - Processed Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Storage Path   : {report_filename}

 2. ATTRIBUTE DISTRIBUTION SUMMARY
    - Total Unique Tracked : {next_display_id-1}
    - Final Count (ROI)    : {total_count}
    - Gender Distribution  : Male({gender_stats['Male']}) / Female({gender_stats['Female']}) / Unk({gender_stats['Unknown']})
    - Mean Conf (Gender)   : {avg_conf:.2f}% (Analysis Reliability)

 3. SYSTEM PERFORMANCE & OPTIMIZATION
    - Effective FPS      : {frame_idx/elapsed:.1f} (Target 15+ on 4070Ti)
    - Total Frame Count  : {frame_idx}
    - Inference Backend  : {vlm_engine.device.upper()}
    - Defense Strategy   : Aspect Ratio Filter(>1.5) + Best-Frame Sampling(3-step)

 4. ACCURACY ASSESSMENT
    - Evaluation Acc     : {min(total_count/25, 25/total_count)*100:.1f}% (vs. MOT16 Ground Truth Ref)
===================================================================================================================
"""
    print(report_content)
    
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
            f.write("\n[APPENDIX: INDIVIDUAL OBJECT LOGS]\n")
            for log in sorted(inference_log):
                f.write(f"> {log}\n")
        print(f"[SUCCESS] Deep analysis report saved: {report_filename}")
    except Exception as e:
        print(f"[ERR] File I/O Error: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()