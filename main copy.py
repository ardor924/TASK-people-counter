import cv2
import time
from collections import defaultdict

import supervision as sv
from ultralytics import YOLO

# =========================
# #1 Config
# =========================
VIDEO_PATH = "data/dev_day.mp4"
MODEL_PATH = "models/yolov8n.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.65      # ID 안정화 핵심
MIN_HITS = 6            # 짧은 트랙 제거

# Counting Zone (화면 하단 기준)
ZONE_START_RATIO = 0.45

# Zone dwell counting
ZONE_DWELL_FRAMES = 12  # 이 프레임 이상 머무르면 1명

# =========================
# #2 Main
# =========================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Video open failed")

    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()

    frame_idx = 0
    start_time = time.time()

    # Tracking statistics
    track_hits = defaultdict(int)
    track_life = defaultdict(int)
    valid_ids = set()
    max_id = 0
    peak_active = 0

    # Zone dwell tracking
    zone_dwell = defaultdict(int)
    counted_ids = set()
    total_count = 0

    print("[INFO] Video started")

    # =========================
    # #3 Frame Loop
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # -------------------------
        # Resize (즉시 프레임 표시)
        # -------------------------
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        display = frame.copy()
        fh, fw = display.shape[:2]

        # Counting zone
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)
        cv2.putText(
            display,
            "COUNT ZONE",
            (10, zone_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # =========================
        # #4 Detection & Tracking
        # =========================
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0:
            detections = detections[detections.confidence >= CONF_THRESH]

        detections = tracker.update_with_detections(detections)

        active_ids = set()

        if detections.tracker_id is not None:
            for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                tid = int(tid)
                max_id = max(max_id, tid)

                track_hits[tid] += 1
                track_life[tid] += 1
                active_ids.add(tid)

                # 짧은 트랙 제거
                if track_hits[tid] < MIN_HITS:
                    continue

                valid_ids.add(tid)

                x1, y1, x2, y2 = map(int, xyxy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # -------------------------
                # Zone Dwell Count (현업식)
                # -------------------------
                if tid not in counted_ids:
                    if cy > zone_y:
                        zone_dwell[tid] += 1
                        if zone_dwell[tid] >= ZONE_DWELL_FRAMES:
                            counted_ids.add(tid)
                            total_count += 1
                    else:
                        zone_dwell[tid] = 0

                # Draw
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"ID {tid}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        peak_active = max(peak_active, len(active_ids))

        # =========================
        # #5 Overlay
        # =========================
        cv2.putText(
            display,
            f"Frame {frame_idx} | Active {len(active_ids)} | Count {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("Tracking", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =========================
    # #6 Result
    # =========================
    elapsed = time.time() - start_time
    fps = frame_idx / elapsed if elapsed > 0 else 0
    avg_life = sum(track_life.values()) / len(track_life) if track_life else 0

    print("=" * 50)
    print(f"[RESULT] TOTAL FRAMES        : {frame_idx}")
    print(f"[RESULT] MAX ID              : {max_id}")
    print(f"[RESULT] VALID IDs           : {len(valid_ids)}")
    print(f"[RESULT] PEAK ACTIVE IDS     : {peak_active}")
    print(f"[RESULT] AVG TRACK LIFE      : {avg_life:.1f} frames")
    print(f"[RESULT] LINE / ZONE COUNT   : {total_count}")
    print(f"[RESULT] EFFECTIVE FPS       : {fps:.1f}")
    print("=" * 50)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
