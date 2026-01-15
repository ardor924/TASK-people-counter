import cv2
import time
from collections import defaultdict
import numpy as np

import supervision as sv
from ultralytics import YOLO

# =========================
# #1 Config
# =========================
VIDEO_PATH = "data/dev_day.mp4"
MODEL_PATH = "models/yolov8n.pt"

RESIZE_WIDTH = 960
CONF_THRESH = 0.5
MIN_HITS = 3
LONG_TRACK_THRESH = 100   # frames

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

    # =========================
    # Tracking statistics
    # =========================
    track_hits = defaultdict(int)
    track_lifetime = defaultdict(int)
    track_history = defaultdict(list)

    valid_ids = set()
    short_tracks = set()
    long_tracks = set()

    max_id = 0
    peak_active = 0
    frames_with_detection = 0

    # =========================
    # Line crossing
    # =========================
    counted_ids = set()
    total_count = 0
    COUNT_LINE_RATIO = 0.6  # 화면 하단 60%

    print("[INFO] Video started")

    # =========================
    # Main Loop
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Resize
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        display = frame.copy()

        frame_h = frame.shape[0]
        count_line_y = int(frame_h * COUNT_LINE_RATIO)

        # Detection
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0:
            detections = detections[detections.confidence >= CONF_THRESH]

        detections = tracker.update_with_detections(detections)

        active_ids = set()

        if detections.tracker_id is not None:
            frames_with_detection += 1

            for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
                tid = int(track_id)
                max_id = max(max_id, tid)

                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                track_hits[tid] += 1
                track_lifetime[tid] += 1
                track_history[tid].append((cx, cy))
                active_ids.add(tid)

                # Short track 판단
                if track_hits[tid] < MIN_HITS:
                    short_tracks.add(tid)
                    continue

                valid_ids.add(tid)

                # Long track 판단
                if track_lifetime[tid] >= LONG_TRACK_THRESH:
                    long_tracks.add(tid)

                # =========================
                # Line Crossing Logic
                # =========================
                if tid not in counted_ids and len(track_history[tid]) >= 2:
                    _, prev_y = track_history[tid][-2]
                    _, curr_y = track_history[tid][-1]

                    if prev_y < count_line_y <= curr_y:
                        counted_ids.add(tid)
                        total_count += 1

                # Draw box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display, f"ID {tid}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        peak_active = max(peak_active, len(active_ids))

        # =========================
        # Overlay
        # =========================
        cv2.line(display, (0, count_line_y), (display.shape[1], count_line_y),
                 (0, 0, 255), 2)

        cv2.putText(
            display,
            f"Frame {frame_idx} | Active {len(active_ids)} | Count {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )

        cv2.imshow("Tracking", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =========================
    # Result Calculation
    # =========================
    elapsed = time.time() - start_time
    fps = frame_idx / elapsed

    lifetimes = list(track_lifetime.values())
    avg_life = np.mean(lifetimes)
    median_life = int(np.median(lifetimes))

    short_ratio = (len(short_tracks) / len(track_lifetime)) * 100
    long_ratio = (len(long_tracks) / len(track_lifetime)) * 100

    id_reuse_ratio = max_id / max(1, len(valid_ids))
    stability_score = peak_active * avg_life / max_id
    unique_person_est = total_count * (avg_life / frame_idx)

    # =========================
    # Result Output
    # =========================
    print("=" * 50)
    print(f"[RESULT] TOTAL FRAMES        : {frame_idx}")
    print(f"[RESULT] MAX ID              : {max_id}")
    print(f"[RESULT] VALID IDs           : {len(valid_ids)}")
    print(f"[RESULT] PEAK ACTIVE IDS     : {peak_active}")
    print(f"[RESULT] AVG TRACK LIFE      : {avg_life:.1f} frames")
    print(f"[RESULT] MEDIAN TRACK LIFE   : {median_life} frames")
    print(f"[RESULT] LONG TRACK %        : {long_ratio:.1f} %")
    print(f"[RESULT] SHORT TRACK %       : {short_ratio:.1f} %")
    print(f"[RESULT] ID REUSE RATIO      : {id_reuse_ratio:.2f}")
    print(f"[RESULT] DETECT FRAMES       : {frames_with_detection}/{frame_idx}")
    print(f"[RESULT] EFFECTIVE FPS       : {fps:.1f}")
    print(f"[RESULT] STABILITY SCORE     : {stability_score:.1f}")
    print(f"[RESULT] UNIQUE PERSON EST.  : {unique_person_est:.1f}")
    print(f"[RESULT] LINE COUNT TOTAL    : {total_count}")
    print("=" * 50)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
