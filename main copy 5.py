import cv2
import time
import numpy as np
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
# #2 Advanced Attribute Logic
# =========================
def analyze_gender_robust(image):
    """
    단순 비율 외에 상체/하체의 색상 대비 및 실루엣을 추가 분석합니다.
    (추후 딥러닝 모델로 교체 시 이 함수 내부만 수정하면 됩니다.)
    """
    try:
        h, w, _ = image.shape
        if h < 40 or w < 20: return "Unknown", 0.0
        
        # 1. 기하학적 분석 (비율)
        ratio = h / w
        
        # 2. 색상 채도 분석 (보통 여성의 의류가 원색/밝은 톤이 많음 - Heuristic)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        
        # 종합 점수 계산 (간이 가중치 방식)
        score = 0
        if ratio > 2.2: score += 1 # 남성 특징 점수
        if avg_saturation > 50: score -= 1 # 여성 특징 점수 (채도가 높음)
        
        gender = "Male" if score >= 0 else "Female"
        confidence = 0.6 # 간이 추론이므로 낮은 신뢰도 부여
        
        return gender, confidence
    except:
        return "Unknown", 0.0

# =========================
# #3 Main
# =========================
def main():
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    loading_img = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(loading_img, "SYSTEM INITIALIZING...", (280, 240), 1, 2, (255, 255, 255), 2)
    cv2.imshow("AI Tracking System", loading_img)
    cv2.waitKey(10)

    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wait_time = max(1, int(1000 / src_fps))
    
    zone_dwell_frames = int(DWELL_TIME_SEC * src_fps)
    id_forget_frames = int(FORGET_TIME_SEC * src_fps)
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * src_fps))

    # State
    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    # 성별 진단용 데이터 구조
    gender_buffer = defaultdict(list) 
    gender_final = {}               
    gender_stats = {"Male": 0, "Female": 0, "Unknown": 0}
    total_analysis_attempts = 0 # 얼마나 분석을 시도했는가

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

        results = model(frame, verbose=False)[0]
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
                        if frame_idx - data["last_frame"] > id_forget_frames:
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

                # -------------------------
                # 신뢰도 기반 성별 투표 (Voting)
                # -------------------------
                if tid not in gender_final:
                    # 최대 20프레임 동안 고화질 크롭 이미지 분석 시도
                    if len(gender_buffer[tid]) < 20:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        res, conf = analyze_gender_robust(crop)
                        if res != "Unknown":
                            gender_buffer[tid].append(res)
                            total_analysis_attempts += 1
                    else:
                        # 과반수 투표로 결정
                        winner = max(set(gender_buffer[tid]), key=gender_buffer[tid].count)
                        gender_final[tid] = winner
                        gender_stats[winner] += 1
                
                label = gender_final.get(tid, f"Analyzing..({len(gender_buffer[tid])}/20)")

                # Count
                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= zone_dwell_frames:
                        counted_ids.add(tid)
                        total_count += 1

                color = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"ID {tid}: {label}", (x1, y1 - 10), 1, 1, color, 2)

        # UI
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"TOTAL COUNT : {total_count}", (20, 40), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 85), 1, 1.2, (0, 255, 255), 2)
        cv2.putText(display, f"FPS : {frame_idx/(time.time()-start_time):.1f}", (20, 125), 1, 1, (150, 150, 150), 1)

        cv2.imshow("AI Tracking System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    # Final Report
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"[DIAGNOSTIC REPORT] ATTRIBUTE ANALYSIS")
    print("=" * 50)
    print(f" 1. GENDER RECOGNITION QUALITY")
    print(f"    - Male : {gender_stats['Male']} / Female : {gender_stats['Female']}")
    print(f"    - Success Rate   : {((gender_stats['Male']+gender_stats['Female'])/(next_display_id-1))*100:.1f}%")
    print(f"    - Avg Attempts   : {total_analysis_attempts/(next_display_id-1):.1f} frames/ID")
    print("-" * 50)
    print(f" 2. COUNTING STABILITY")
    print(f"    - Final Count    : {total_count}")
    print(f"    - Accuracy (Rel) : {min(total_count/25, 25/total_count)*100:.1f}%")
    print("=" * 50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()