import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict, Counter
from datetime import datetime

# =========================
# #1 Global Policy (Low-Spec / Edge Mode)
# =========================
# VIDEO_PATH = "data/dev_day.mp4"
# VIDEO_PATH = "data/eval_night.mp4"
VIDEO_PATH = "data/eval_indoors.mp4"

DETECTOR_PATH = "models/yolov8n.pt"
CLASSIFIER_PATH = "models/yolov8n-cls.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.55       
DWELL_TIME_SEC = 0.5     
FORGET_TIME_SEC = 0.7    
TRACK_BUFFER_SEC = 2.0   
ZONE_START_RATIO = 0.45

# =========================
# #2 Hybrid Attribute Engine (Voting Capable)
# =========================
class HybridAttributeEngine:
    def __init__(self, device="cuda"):
        from ultralytics import YOLO
        self.device = device
        
        # 1. 성별 분류 모델 (YOLOv8-cls)
        try:
            self.classifier = YOLO(CLASSIFIER_PATH).to(self.device)
            print(f"[INFO] Gender Classifier Loaded on {self.device}.")
        except Exception as e:
            print(f"[WARN] Failed to load classifier: {e}")
            self.classifier = None

        # 2. 색상 분석 엔진 (Advanced HSV)
        self.color_centers = {
            "Black":  [0, 0, 30],
            "White":  [0, 0, 220],
            "Red":    [0, 200, 150],
            "Blue":   [115, 200, 150],
            "Yellow": [25, 200, 200],
            "Gray":   [0, 0, 100],
            "Green":  [60, 150, 100]
        }

    def analyze(self, crop_cv2):
        # 기본 반환값
        res_g, g_conf = "Unk", 0.0
        res_c, c_conf = "Unk", 0.0

        if crop_cv2.size == 0:
            return {"g": res_g, "c": res_c, "g_cf": 0, "c_cf": 0}

        # [Step 1] 성별 분석
        if self.classifier:
            try:
                res = self.classifier(crop_cv2, verbose=False, device=self.device)[0]
                top1_idx = res.probs.top1
                conf = res.probs.top1conf.item() * 100
                
                # 클래스 이름 기반 매핑 (안전장치)
                class_name = res.names[top1_idx].lower()
                if 'woman' in class_name or 'female' in class_name:
                    res_g = 'Female'
                elif 'man' in class_name or 'male' in class_name:
                    res_g = 'Male'
                else:
                    res_g = "Male" if top1_idx % 2 != 0 else "Female"
                g_conf = conf
            except: pass

        # [Step 2] 색상 분석 (K-Means)
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
                    
                    # 피부색 필터링
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
    import torch
    import supervision as sv
    from ultralytics import YOLO

    for folder in ["logs", "best_samples"]:
        if not os.path.exists(folder): os.makedirs(folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running in Hybrid Voting Mode (Device: {device})")

    detector = YOLO(DETECTOR_PATH).to(device)
    hybrid_engine = HybridAttributeEngine(device=device)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wait_time = max(1, int(1000 / src_fps))
    
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * src_fps))

    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    # [데이터 저장소]
    # voting_box: {tid: {'g_scores': {'Male': 0, 'Female': 0}, 'c_list': [], 'best_img': None, 'max_conf': 0}}
    voting_box = defaultdict(lambda: {'g_scores': defaultdict(float), 'c_list': [], 'best_img': None, 'max_conf': 0})
    id_attributes = {} # 최종 확정된 결과
    inference_log = []
    
    gender_stats = {"Male": 0, "Female": 0, "Unk": 0}
    
    frame_idx, start_time = 0, time.time()
    
    cv2.namedWindow("Hybrid Low-Spec AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid Low-Spec AI", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
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

        # 1. Detection
        results = detector(frame, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. Filtering
        valid_indices = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            aspect_ratio = bh / bw if bw > 0 else 0
            if aspect_ratio > 1.4 and detections.confidence[i] >= CONF_THRESH:
                valid_indices.append(i)
        
        detections = detections[valid_indices]
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()

        if detections.tracker_id is not None:
            for xyxy, raw_id, conf in zip(detections.xyxy, detections.tracker_id, detections.confidence):
                raw_id = int(raw_id)
                track_hits[raw_id] += 1
                if track_hits[raw_id] < 5: continue 

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
                
                # 3. Hybrid Voting Analysis (10, 20, 30 프레임에서 수집)
                if tid not in id_attributes:
                    if track_hits[raw_id] in [10, 20, 30]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = hybrid_engine.analyze(crop)
                            
                            # (1) 성별: 확신도 가중치 누적 (Weighted Voting)
                            if res['g'] in ['Male', 'Female']:
                                voting_box[tid]['g_scores'][res['g']] += res['g_cf']
                            
                            # (2) 색상: 리스트에 추가 (Majority Voting)
                            if res['c'] != "Unk":
                                voting_box[tid]['c_list'].append(res['c'])
                            
                            # (3) Best Image: 합산 점수가 높은 순간의 이미지를 임시 저장
                            curr_score = res['g_cf'] + res['c_cf']
                            if curr_score > voting_box[tid]['max_conf']:
                                voting_box[tid]['max_conf'] = curr_score
                                voting_box[tid]['best_img'] = crop.copy()

                    # 31프레임 시점에 "최종 확정" (Decision Making)
                    if track_hits[raw_id] == 31:
                        vdata = voting_box[tid]
                        
                        # 성별 결정: 누적 점수가 높은 쪽
                        m_score = vdata['g_scores']['Male']
                        f_score = vdata['g_scores']['Female']
                        
                        final_g = "Unk"
                        # 최소 1번이라도 유의미한 탐지가 있었고, 합산 점수가 일정 이상이어야 함
                        if (m_score + f_score) > 50: 
                            final_g = "Male" if m_score >= f_score else "Female"
                        
                        # 색상 결정: 최빈값 (Most Common)
                        final_c = "Unk"
                        if vdata['c_list']:
                            final_c = Counter(vdata['c_list']).most_common(1)[0][0]
                        
                        # 결과 확정 및 통계 반영
                        id_attributes[tid] = {"g": final_g, "c": final_c}
                        gender_stats[final_g] += 1
                        
                        # 로그 기록
                        log_conf = max(m_score, f_score) / 3 # 평균 근사치
                        inference_log.append(f"[ID {tid:02d}] Final: {final_g}({log_conf:.0f}) | {final_c}")
                        
                        # Best Sample 최종 저장 (voting_box에 저장된 최고 프레임 사용)
                        if vdata['best_img'] is not None:
                            img_filename = f"LOW_ID{tid:02d}_{final_g}_{final_c}.jpg"
                            # 임시로 메모리에 들고 있다가 나중에 저장하거나 바로 저장 가능
                            # 여기서는 리스트에 넣어서 마지막에 일괄 저장하도록 구조 유지
                            voting_box[tid]['final_label'] = f"{final_g}_{final_c}"

                # UI 표시 (분석 중일 땐 Analyzing 표시)
                attr = id_attributes.get(tid, {"g": "Scaning..", "c": "."})
                label = f"ID:{tid} | {attr['g'][0]}/{attr['c']}"

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * src_fps):
                        counted_ids.add(tid)
                        total_count += 1

                # 색상: 분석 완료(Green), 진행 중(White)
                color_box = (0, 255, 0) if tid in id_attributes else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.7, color_box, 1)

        # UI Overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"HYBRID VOTING SYSTEM: {os.path.basename(VIDEO_PATH)}", (20, 45), 1, 1.4, (0, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        
        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(display, f"PROG : {int(progress)}% | FPS : {frame_idx/(time.time()-start_time):.1f}", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("Hybrid Low-Spec AI", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    elapsed = time.time() - start_time
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    report_filename = f"logs/report_LOW_{video_base_name}_{timestamp}.txt"

    # [1] Best Samples 저장 (Top 5)
    print("\n[INFO] Saving Top 5 Best Samples...")
    # 점수 기준 정렬 (voting_box의 max_conf 활용)
    sorted_votes = sorted(voting_box.items(), key=lambda x: x[1]['max_conf'], reverse=True)[:5]
    
    for rank, (tid, data) in enumerate(sorted_votes, 1):
        if data['best_img'] is not None:
            try:
                label = data.get('final_label', 'Analyzing')
                img_filename = f"LOW_{video_base_name}_{timestamp}_Top{rank}_ID{tid:02d}_{label}.jpg"
                cv2.imwrite(os.path.join("best_samples", img_filename), data['best_img'])
                print(f"   > Saved: {img_filename}")
            except Exception as e:
                print(f"   > Error saving ID {tid}: {e}")

    # [2] 리포트 생성
    report_content = f"""
===================================================================================================================
[FINAL HYBRID SYSTEM REPORT (VOTING ENABLED)]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : {os.path.basename(VIDEO_PATH)}
    - Processed Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Storage Path   : {report_filename}

 2. PERFORMANCE SUMMARY
    - Total Tracked  : {next_display_id-1}
    - Final Count    : {total_count}
    - Gender Stats   : Male({gender_stats['Male']}) / Female({gender_stats['Female']}) / Unk({gender_stats['Unk']})
    - Effective FPS  : {frame_idx/elapsed:.1f}
    
 3. CONFIGURATION
    - Mode           : Low-Spec / Hybrid Voting
    - Logic          : 3-Frame Temporal Voting (Majority Rule)
===================================================================================================================
"""
    print(report_content)
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
            f.write("\n[APPENDIX: INDIVIDUAL LOGS]\n")
            for log in sorted(inference_log): f.write(f"> {log}\n")
    except: pass

    # [3] 요약 UI
    summary_bg = np.zeros((450, 650, 3), dtype=np.uint8)
    summary_bg[:] = (35, 35, 35)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(summary_bg, "HYBRID VOTING SUMMARY", (50, 50), font, 1, (255, 255, 0), 2)
    cv2.line(summary_bg, (50, 65), (600, 65), (150, 150, 150), 1)
    
    cv2.putText(summary_bg, f"Video: {os.path.basename(VIDEO_PATH)}", (50, 110), font, 0.7, (255, 255, 255), 1)
    cv2.putText(summary_bg, f"Total Count: {total_count}", (50, 160), font, 0.9, (0, 255, 0), 2)
    cv2.putText(summary_bg, f"Male: {gender_stats['Male']} / Female: {gender_stats['Female']}", (50, 210), font, 0.7, (255, 255, 255), 1)
    cv2.putText(summary_bg, f"Avg FPS: {frame_idx/elapsed:.1f}", (50, 260), font, 0.7, (255, 255, 255), 1)
    cv2.putText(summary_bg, f"Logic: Temporal Voting (3-Step)", (50, 310), font, 0.6, (200, 200, 0), 1)
    
    cv2.putText(summary_bg, "Press any key to exit.", (50, 410), font, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Final Summary Report", summary_bg)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()