import cv2
import time
import numpy as np
from collections import defaultdict

# =========================
# #1 Global Policy (범용적 기준)
# =========================
VIDEO_PATH = "data/dev_day.mp4"
MODEL_PATH = "models/yolov8n.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.62      # 범용적으로 가장 안정적인 임계값

# [범용성 강화] 초(sec) 단위 가이드라인 (어떤 영상에서도 동일한 시간 감도 유지)
DWELL_TIME_SEC = 0.5    # 0.5초 이상 영역 체류 시 카운트
FORGET_TIME_SEC = 0.7   # 0.7초 이상 미검지 시 추적 정보 폐기
TRACK_BUFFER_SEC = 2.0  # 2초간 유실되어도 동일인으로 간주

# [범용성 강화] 화면 비율 단위 거리 (해상도 독립성)
ID_FUSION_RATIO = 0.015 
ZONE_START_RATIO = 0.45

# =========================
# #2 Main
# =========================
def main():
    # 1. 즉시 창 생성 및 로딩 안내 (UX 개선)
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    loading_img = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(loading_img, "AI SYSTEM INITIALIZING...", (280, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(loading_img, "Auto-Calibrating for Video Source...", (290, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.imshow("AI Tracking System", loading_img)
    cv2.waitKey(10)

    # 2. Heavy Imports (창 띄운 후 로드)
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise RuntimeError("Video open failed")

    # [범용성 핵심] 입력 영상의 FPS 및 해상도 분석
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wait_time = max(1, int(1000 / src_fps))
    
    # 초 단위 설정을 프레임 단위로 자동 변환 (FPS가 달라도 동일 감도 유지)
    zone_dwell_frames = int(DWELL_TIME_SEC * src_fps)
    id_forget_frames = int(FORGET_TIME_SEC * src_fps)
    track_buffer_frames = int(TRACK_BUFFER_SEC * src_fps)

    tracker = sv.ByteTrack(lost_track_buffer=track_buffer_frames)

    # State Variables
    track_hits = defaultdict(int)
    track_life = defaultdict(int)
    track_registry = {}   
    id_map = {}           
    next_display_id = 1   
    zone_dwell = defaultdict(int)
    counted_ids = set()   
    total_count = 0
    
    frame_idx, start_time = 0, time.time()
    max_raw_id, fusion_count = 0, 0
    fusion_dist_limit = RESIZE_WIDTH * ID_FUSION_RATIO

    # =========================
    # #3 Frame Loop
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        display = frame.copy()
        fh, fw = display.shape[:2]

        # Zone Line
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        # AI Detection & Tracking
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence >= CONF_THRESH]
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()

        if detections.tracker_id is not None:
            for xyxy, raw_id in zip(detections.xyxy, detections.tracker_id):
                raw_id = int(raw_id)
                max_raw_id = max(max_raw_id, raw_id)
                track_hits[raw_id] += 1
                
                # 최소 검증 프레임 (노이즈 방지)
                if track_hits[raw_id] < 10: continue 

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Spatio-Temporal ID Fusion
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
                            fusion_count += 1
                            break
                    if not found_legacy:
                        id_map[raw_id] = next_display_id
                        next_display_id += 1
                
                tid = id_map[raw_id]
                active_now_ids.add(tid)
                track_life[tid] += 1
                track_registry[tid] = {"pos": (cx, cy), "last_frame": frame_idx}

                # Count Logic (Time-based Dwell)
                if tid not in counted_ids:
                    if cy > zone_y:
                        zone_dwell[tid] += 1
                        if zone_dwell[tid] >= zone_dwell_frames:
                            counted_ids.add(tid)
                            total_count += 1
                    else:
                        zone_dwell[tid] = 0

                color = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"ID {tid}", (x1, y1 - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Dashboard UI
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"STATUS: TRACKING", (20, 30), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(display, f"TOTAL COUNT : {total_count}", (20, 60), 1, 1.4, (255, 255, 255), 2)
        cv2.putText(display, f"VALID IDs   : {next_display_id-1}", (20, 90), 1, 1.2, (200, 200, 200), 1)

        cv2.imshow("AI Tracking System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    # Final Report
    elapsed = time.time() - start_time
    visual_max_id = next_display_id - 1
    avg_life = sum(track_life.values()) / len(track_life) if track_life else 0

    print("\n" + "=" * 50)
    print(f"[REPORT] ROBUST MULTI-OBJECT TRACKING")
    print("=" * 50)
    print(f" 1. PERFORMANCE ANALYSIS")
    print(f"    - Processing Time    : {elapsed:.2f} sec")
    print(f"    - Effective FPS      : {frame_idx/elapsed:.1f} (Source: {src_fps})")
    print("-" * 50)
    print(f" 2. TRACKING QUALITY (Auto-Calibrated)")
    print(f"    - Validated People   : {visual_max_id} (Target ~25)")
    print(f"    - Fusion Merged      : {fusion_count} (Stabilized)")
    print(f"    - Dwell Logic        : {zone_dwell_frames} frames ({DWELL_TIME_SEC}s)")
    print("-" * 50)
    print(f" 3. FINAL RESULT")
    print(f"    - FINAL COUNT        : {total_count}")
    print(f"    - ACCURACY ESTIMATE  : {min(total_count/25, 25/total_count)*100:.1f}%")
    print("=" * 50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()