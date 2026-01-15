import cv2
import time
import numpy as np
from collections import defaultdict

# =========================
# #1 Config
# =========================
VIDEO_PATH = "data/dev_day.mp4"
MODEL_PATH = "models/yolov8n.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.62      
MIN_HITS = 10           
TRACK_BUFFER = 60       

ID_FUSION_RATIO = 0.015 
ID_FORGET_FRAMES = 20   

ZONE_START_RATIO = 0.45
ZONE_DWELL_FRAMES = 15  

# =========================
# #2 Main
# =========================
def main():
    # -------------------------
    # 1. 즉시 창 생성 (Start UI First)
    # -------------------------
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    # 로딩 배경 생성
    loading_img = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(loading_img, "AI SYSTEM INITIALIZING...", (280, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(loading_img, "Loading Neural Networks & Tracker...", (310, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    cv2.imshow("AI Tracking System", loading_img)
    cv2.waitKey(10) # 창이 운영체제에서 뜰 수 있게 최소한의 대기

    # -------------------------
    # 2. Heavy Imports & Model Init
    # -------------------------
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack(lost_track_buffer=TRACK_BUFFER)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise RuntimeError("Video open failed")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wait_time = int(1000 / video_fps)

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
    max_raw_id = 0
    fusion_count = 0
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

        # Counting Zone UI
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
                if track_hits[raw_id] < MIN_HITS: continue 

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ID Fusion Logic
                if raw_id not in id_map:
                    found_legacy = False
                    for disp_id, data in list(track_registry.items()):
                        if disp_id in active_now_ids: continue 
                        if frame_idx - data["last_frame"] > ID_FORGET_FRAMES:
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

                # Counting Dwell Logic
                if tid not in counted_ids:
                    if cy > zone_y:
                        zone_dwell[tid] += 1
                        if zone_dwell[tid] >= ZONE_DWELL_FRAMES:
                            counted_ids.add(tid)
                            total_count += 1
                    else:
                        zone_dwell[tid] = 0

                # Box Draw
                color = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"ID {tid}", (x1, y1 - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # -------------------------
        # Dashboard UI Overlay
        # -------------------------
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        cv2.putText(display, f"SYSTEM STATUS: ONLINE", (20, 30), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(display, f"TOTAL COUNT   : {total_count}", (20, 60), 1, 1.4, (255, 255, 255), 2)
        cv2.putText(display, f"VALIDATED IDs : {next_display_id-1}", (20, 90), 1, 1.2, (200, 200, 200), 1)

        cv2.imshow("AI Tracking System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    # =========================
    # #4 Final 풍성한 리포트 (Debug & Evaluation)
    # =========================
    elapsed = time.time() - start_time
    visual_max_id = next_display_id - 1
    avg_life = sum(track_life.values()) / len(track_life) if track_life else 0

    print("\n" + "=" * 50)
    print(f"[REPORT] ROBUST TRACKING SUMMARY")
    print("=" * 50)
    print(f" 1. PERFORMANCE ANALYSIS")
    print(f"    - Total Frames       : {frame_idx}")
    print(f"    - Processing Time    : {elapsed:.2f} sec")
    print(f"    - Effective FPS      : {frame_idx/elapsed:.1f}")
    print("-" * 50)
    print(f" 2. TRACKING QUALITY")
    print(f"    - Raw Tracker IDs    : {max_raw_id} (All Detected)")
    print(f"    - Validated People   : {visual_max_id} (Target ~25)")
    print(f"    - Fusion Merged      : {fusion_count} (Stabilized)")
    print(f"    - Avg Persistence    : {avg_life:.1f} frames")
    print("-" * 50)
    print(f" 3. FINAL RESULT")
    print(f"    - FINAL COUNT        : {total_count}")
    print(f"    - ACCURACY (Rel. 25) : {min(total_count/25, 25/total_count)*100:.1f}%")
    print("=" * 50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()