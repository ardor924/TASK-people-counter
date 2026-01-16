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
VIDEO_PATH = "data/dev_day.mp4"
# VIDEO_PATH = "data/eval_night.mp4"
# VIDEO_PATH = "data/eval_indoors.mp4"

DETECTOR_PATH = "models/yolov8n.pt"
# CLASSIFIER_PATH 제거 (속도 최적화)

RESIZE_WIDTH = 960
CONF_THRESH = 0.50       # 탐지율 확보를 위해 약간 낮춤
DWELL_TIME_SEC = 0.5     
FORGET_TIME_SEC = 0.7    
TRACK_BUFFER_SEC = 2.0   
ZONE_START_RATIO = 0.45

# =========================
# #2 Lightweight CV Engine (HSV + K-Means)
# =========================
class LightweightAttributeEngine:
    def __init__(self):
        # 색상 맵 (HSV 범위)
        self.color_map = {
            "Black":  ([0, 0, 0], [180, 255, 50]),
            "White":  ([0, 0, 200], [180, 30, 255]),
            "Red":    ([0, 100, 100], [10, 255, 255]),
            "Blue":   ([100, 100, 100], [130, 255, 255]),
            "Yellow": ([20, 100, 100], [30, 255, 255]),
            "Gray":   ([0, 0, 50], [180, 50, 200])
        }
        print(f"[INFO] Lightweight CV Engine Active (Color Only / Fast Mode).")

    def analyze(self, crop_cv2):
        try:
            # 1. 성별: 저사양 모드에서는 과감히 생략 (리소스 절약)
            gender = "N/A" 
            
            # 2. 색상 분석 (HSV + K-Means)
            h, w = crop_cv2.shape[:2]
            
            # [핵심] ROI를 좁게 설정하여 배경 노이즈 제거 (가슴 중앙 집중)
            # 높이: 상단 15%~45% (어깨~가슴)
            # 너비: 중앙 30%~70% (팔/배경 제외)
            upper_body = crop_cv2[int(h*0.15):int(h*0.45), int(w*0.3):int(w*0.7)]
            
            color = "Unk"
            if upper_body.size > 0:
                data = upper_body.reshape((-1, 3)).astype(np.float32)
                
                # K-Means로 가장 지배적인 색상 1개 추출
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                dominant_hsv = cv2.cvtColor(np.uint8([[centers[0]]]), cv2.COLOR_BGR2HSV)[0][0]
                
                # HSV 거리 기반 매칭 (가장 가까운 색상 찾기)
                best_dist = 9999
                for name, (lower, upper) in self.color_map.items():
                    # 범위 내 포함 여부 체크 + 중심점 거리 계산 병행
                    lower_np = np.array(lower)
                    upper_np = np.array(upper)
                    
                    # 단순 범위 체크
                    if np.all(lower_np <= dominant_hsv) and np.all(dominant_hsv <= upper_np):
                        color = name
                        break
                    
                    # 범위 밖이라면 중심점과의 거리로 근사 (Fallback)
                    center_color = (lower_np + upper_np) / 2
                    dist = np.linalg.norm(dominant_hsv - center_color)
                    if dist < best_dist:
                        best_dist = dist
                        color = name

            return {"g": gender, "c": color}
        except:
            return {"g": "N/A", "c": "Unk"}

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

    # 장치 설정 (Detection은 GPU가 있으면 씀)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running in Low-Spec Mode (Device: {device})")

    # Detection 모델만 로드 (가벼움)
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
    best_crops_store = {} # 이미지 저장용 (Detection Confidence 기준)
    
    frame_idx, start_time = 0, time.time()
    
    cv2.namedWindow("Low-Spec AI System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Low-Spec AI System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
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

        # 1. Detection (YOLOv8n)
        results = detector(frame, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. Filtering (Aspect Ratio)
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
                
                # 저사양은 추적 시작을 조금 더 빠르게
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
                        # 매핑 거리 제한
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
                    if track_hits[raw_id] == 15: # 15프레임째에 딱 한번 분석
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = cv_engine.analyze(crop)
                            id_attributes[tid] = res
                            
                            # 로그 기록 (Detection Confidence 사용)
                            det_conf = float(conf) * 100
                            inference_log.append(f"[ID {tid:02d}] Det Conf: {det_conf:.1f}% | Color: {res['c']}")
                            
                            # Best Sample 저장 (탐지 신뢰도가 높은 순)
                            best_crops_store[tid] = {
                                'conf': det_conf,
                                'img': crop.copy(),
                                'label': f"{res['c']}"
                            }

                attr = id_attributes.get(tid, {"c": "..."})
                label = f"ID:{tid} | {attr['c']}"

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
        cv2.putText(display, f"LOW-SPEC MODE (Color Only)", (20, 45), 1, 1.5, (0, 255, 255), 2)
        cv2.putText(display, f"COUNT : {total_count}", (20, 95), 1, 1.5, (0, 255, 0), 2)
        
        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(display, f"PROG : {int(progress)}% | FPS : {frame_idx/(time.time()-start_time):.1f}", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("Low-Spec AI System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    elapsed = time.time() - start_time
    
    # 시간 기반 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    report_filename = f"logs/report_LOW_{video_base_name}_{timestamp}.txt"

    # [1] Best Samples 저장 (Detection Confidence 기준)
    print("\n[INFO] Saving Top 5 Best Samples...")
    sorted_crops = sorted(best_crops_store.items(), key=lambda item: item[1]['conf'], reverse=True)[:5]
    
    for rank, (tid, data) in enumerate(sorted_crops, 1):
        try:
            img_filename = f"LOW_{video_base_name}_{timestamp}_Top{rank}_ID{tid:02d}_{data['label']}.jpg"
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
    - Attributes     : Color Only (HSV K-Means)
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
    cv2.putText(summary_bg, f"Optimization: Color Only (Fast)", (50, 230), font, 0.6, (200, 200, 0), 1)
    
    cv2.putText(summary_bg, "Logs saved. Press any key to exit.", (50, 330), font, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Final Summary Report", summary_bg)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()