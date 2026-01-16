import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime

# =========================
# #1 Global Policy (Low-Spec / Edge Mode)
# =========================
# VIDEO_PATH = "data/dev_day.mp4"
# VIDEO_PATH = "data/eval_night.mp4"
VIDEO_PATH = "data/eval_indoors.mp4"

DETECTOR_PATH = "models/yolov8n.pt"
# CLASSIFIER_PATH 제거 (속도 및 최적화)

RESIZE_WIDTH = 960
CONF_THRESH = 0.55       # 정확도 향상을 위해 소폭 상향
DWELL_TIME_SEC = 0.5     
FORGET_TIME_SEC = 0.7    
TRACK_BUFFER_SEC = 2.0   
ZONE_START_RATIO = 0.45

# =========================
# #2 Advanced CV Engine (Robust Color Extraction)
# =========================
class LightweightAttributeEngine:
    def __init__(self):
        # 고도화된 색상 중심점 (HSV 기준) - 조명 변화 대응
        # Format: [Hue(0-180), Sat(0-255), Val(0-255)]
        self.color_centers = {
            "Black":  [0, 0, 30],
            "White":  [0, 0, 220],
            "Red":    [0, 200, 150],   # Red는 0과 180 양쪽 분포 (로직에서 처리)
            "Blue":   [115, 200, 150],
            "Yellow": [25, 200, 200],
            "Gray":   [0, 0, 100],     # Saturation이 낮고 Value가 중간
            "Green":  [60, 150, 100]
        }
        print(f"[INFO] Advanced CV Engine Active (Multi-Cluster K-Means).")

    def analyze(self, crop_cv2):
        try:
            h, w = crop_cv2.shape[:2]
            # [전략 1] ROI 정밀화: 어깨와 배를 제외하고 '가슴 중앙'만 타격
            roi = crop_cv2[int(h*0.2):int(h*0.5), int(w*0.3):int(w*0.7)]
            
            if roi.size == 0: return {"g": "N/A", "c": "Unk", "c_cf": 0}

            # [전략 2] K-Means (K=3): 옷, 피부, 그림자를 분리
            pixels = roi.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
            # 3개의 주요 색상 추출
            K = 3
            if len(pixels) < K: K = 1
            _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            
            # 각 클러스터의 비중 계산
            counts = np.bincount(labels.flatten())
            total_pixels = len(roi.reshape((-1, 3)))

            best_color = "Unk"
            max_score = -1.0 # (비중 * (1-거리)) 점수

            for i, center in enumerate(centers):
                weight = counts[i] / total_pixels
                
                # 너무 작은 노이즈 클러스터(15% 미만) 무시
                if weight < 0.15: continue

                c_bgr = np.uint8([[center]])
                c_hsv = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2HSV)[0][0]
                
                # [전략 3] 피부색 필터링 (동양인/백인 피부톤 대략적 범위)
                # Hue 5~25, Sat 50~180 범위는 옷이 아닐 확률 높음 -> 점수 패널티
                is_skin = (5 <= c_hsv[0] <= 25) and (40 <= c_hsv[1] <= 180)
                
                # 색상 매칭 (Weighted Distance)
                min_dist = 9999
                matched_name = "Unk"
                
                for name, target in self.color_centers.items():
                    # Hue 차이 계산 (Red 0, 180 경계 처리)
                    h_diff = abs(float(c_hsv[0]) - target[0])
                    if h_diff > 90: h_diff = 180 - h_diff
                    
                    # 거리 가중치: Hue(색상) 2.5배, Sat(채도) 1.0배, Val(명도) 0.8배
                    # 명도는 조명에 따라 바뀌므로 가중치를 낮춤
                    dist = np.sqrt(2.5 * (h_diff**2) + 
                                   1.0 * ((float(c_hsv[1]) - target[1])**2) + 
                                   0.8 * ((float(c_hsv[2]) - target[2])**2))
                    
                    if dist < min_dist:
                        min_dist = dist
                        matched_name = name
                
                # 점수 산출: (거리 역수) * (가중치) 
                # 피부색이면 점수 대폭 깎음
                final_score = (1000 - min_dist) * weight
                if is_skin: final_score *= 0.3 

                if final_score > max_score:
                    max_score = final_score
                    best_color = matched_name
                    
                    # 확신도(Confidence) 정규화 (0~100%)
                    final_conf = max(0, min(100, (1 - min_dist/300) * 100))

            return {"g": "N/A", "c": best_color, "c_cf": final_conf}
        except:
            return {"g": "N/A", "c": "Unk", "c_cf": 0}

# =========================
# #3 Main Pipeline
# =========================
def main():
    import torch
    import supervision as sv
    from ultralytics import YOLO

    # 필수 폴더 생성
    for folder in ["logs", "best_samples"]:
        if not os.path.exists(folder): os.makedirs(folder)

    # 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running in Low-Spec Mode (Device: {device})")

    # Detection 모델 로드
    detector = YOLO(DETECTOR_PATH).to(device)
    cv_engine = LightweightAttributeEngine()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wait_time = max(1, int(1000 / src_fps))
    
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * src_fps))

    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    # 데이터 저장소
    id_attributes = {}
    inference_log = []
    best_crops_store = {} 
    
    frame_idx, start_time = 0, time.time()
    
    cv2.namedWindow("Advanced Low-Spec AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Advanced Low-Spec AI", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
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
                
                # 3. Fast Analysis (단 1회 수행)
                if tid not in id_attributes:
                    if track_hits[raw_id] == 15:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = cv_engine.analyze(crop)
                            id_attributes[tid] = res
                            
                            det_conf = float(conf) * 100
                            # 로그에 색상 확신도 포함
                            inference_log.append(f"[ID {tid:02d}] Det: {det_conf:.0f}% | Color: {res['c']} ({res['c_cf']:.0f}%)")
                            
                            # Best Sample 저장
                            best_crops_store[tid] = {
                                'conf': res['c_cf'], # 색상 확신도 기준 저장
                                'img': crop.copy(),
                                'label': f"{res['c']}"
                            }

                attr = id_attributes.get(tid, {"c": "...", "c_cf": 0})
                label = f"ID:{tid} | {attr['c']}" # UI 간소화

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * src_fps):
                        counted_ids.add(tid)
                        total_count += 1

                color_box = (0, 255, 0) if tid in counted_ids else (200, 200, 200)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.8, color_box, 1)

        # UI Overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"LOW-SPEC MODE (Robust Color)", (20, 45), 1, 1.5, (0, 255, 255), 2)
        cv2.putText(display, f"COUNT : {total_count}", (20, 95), 1, 1.5, (0, 255, 0), 2)
        
        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(display, f"PROG : {int(progress)}% | FPS : {frame_idx/(time.time()-start_time):.1f}", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("Advanced Low-Spec AI", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    elapsed = time.time() - start_time
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    report_filename = f"logs/report_LOW_{video_base_name}_{timestamp}.txt"

    # [1] Best Samples 저장
    print("\n[INFO] Saving Top 5 Best Samples...")
    sorted_crops = sorted(best_crops_store.items(), key=lambda item: item[1]['conf'], reverse=True)[:5]
    
    for rank, (tid, data) in enumerate(sorted_crops, 1):
        try:
            img_filename = f"LOW_{video_base_name}_{timestamp}_Top{rank}_ID{tid:02d}_{data['label']}_{data['conf']:.0f}pct.jpg"
            save_path = os.path.join("best_samples", img_filename)
            cv2.imwrite(save_path, data['img'])
            print(f"   > Saved: {img_filename}")
        except Exception as e:
            print(f"   > Failed to save sample ID {tid}: {e}")

    # [2] 리포트 생성
    report_content = f"""
===================================================================================================================
[FINAL LOW-SPEC SYSTEM REPORT]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : {os.path.basename(VIDEO_PATH)}
    - Processed Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Storage Path   : {report_filename}

 2. PERFORMANCE SUMMARY
    - Total Tracked  : {next_display_id-1}
    - Final Count    : {total_count}
    - Effective FPS  : {frame_idx/elapsed:.1f}
    
 3. CONFIGURATION
    - Mode           : Low-Spec / Edge Optimization
    - Attributes     : Advanced Color (Multi-KMeans + Weighted HSV)
    - Gender Logic   : Disabled (Resource Saving)
===================================================================================================================
"""
    print(report_content)
    
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
            f.write("\n[APPENDIX: DETECTED OBJECT COLORS]\n")
            for log in sorted(inference_log):
                f.write(f"> {log}\n")
        print(f"[SUCCESS] Report saved: {report_filename}")
    except:
        pass

    # [3] 요약 UI 표시
    summary_bg = np.zeros((400, 600, 3), dtype=np.uint8)
    summary_bg[:] = (30, 30, 30)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    green = (0, 255, 0)
    cyan = (255, 255, 0)
    
    cv2.putText(summary_bg, "LOW-SPEC SYSTEM SUMMARY", (50, 50), font, 1, cyan, 2)
    cv2.line(summary_bg, (50, 65), (550, 65), (150, 150, 150), 1)
    
    cv2.putText(summary_bg, f"Video: {os.path.basename(VIDEO_PATH)}", (50, 110), font, 0.6, white, 1)
    cv2.putText(summary_bg, f"Total Count: {total_count}", (50, 150), font, 0.8, green, 2)
    cv2.putText(summary_bg, f"Avg FPS: {frame_idx/elapsed:.1f}", (50, 190), font, 0.6, white, 1)
    cv2.putText(summary_bg, f"Optimization: K-Means Clustering", (50, 230), font, 0.6, (200, 200, 0), 1)
    
    cv2.putText(summary_bg, "Logs saved. Press any key to exit.", (50, 330), font, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Final Summary Report", summary_bg)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()