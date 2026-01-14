import cv2
import argparse
import os
import time
import threading
from datetime import datetime

from src.core import AIEngine, AttributeResolver
from src.ui import UIManager

g_loading_complete = False
ai_engine = None

def load_ai_task():
    global ai_engine, g_loading_complete
    try:
        model_path = os.path.join("models", "yolov8n.pt")
        # 라인 위치: 0.55
        ai_engine = AIEngine(model_path, line_pos=0.55)
        g_loading_complete = True
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)

def main(source):
    global g_loading_complete, ai_engine

    ui = UIManager(display_size=(960, 540))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Video not found")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    delay_ms = int(1000 / fps)

    # UI 로딩 (즉시)
    ret, first_frame = cap.read()
    if ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        loading_img = ui.draw_loading(first_frame)
        cv2.imshow("People Counter", loading_img)
        cv2.waitKey(1)

    # AI 로딩 (백그라운드)
    threading.Thread(target=load_ai_task, daemon=True).start()

    dots = ""
    last_tick = time.time()
    while not g_loading_complete:
        if time.time() - last_tick > 0.5:
            dots = "." * ((len(dots)+1)%4)
            last_tick = time.time()
            loading_img = ui.draw_loading(first_frame, dots)
            cv2.imshow("People Counter", loading_img)
            if cv2.waitKey(100) & 0xFF == ord('q'): return

    # 준비 완료
    # [수정] supervision line_zone 객체 자체를 리턴받음
    sv_line_zone = ai_engine.init_line_zone(orig_w, orig_h)
    resolver = AttributeResolver()
    
    log_data = []
    max_density = 0
    frame_cnt = 0
    paused = False
    session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("✅ System Ready.")

    while True:
        loop_start = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        
        if paused:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret: break
        frame_cnt += 1

        # A. AI Process
        detections, cross_in, cross_out = ai_engine.process_frame(frame)

        # B. Analysis
        labels = []
        for i, (xyxy, mask, conf, class_id, track_id, data) in enumerate(detections):
            if track_id is None: 
                labels.append("")
                continue

            attr = resolver.resolve(frame, xyxy, track_id)
            label = f"ID:{track_id}|{attr['gender'][0]}({int(attr['g_conf']*100)}%)|{attr['color']}({int(attr['c_conf']*100)}%)"
            labels.append(label)

            direction = None
            if len(cross_in) > i and cross_in[i]: direction = "IN"
            elif len(cross_out) > i and cross_out[i]: direction = "OUT"
            
            if direction:
                log_data.append({
                    "Frame": frame_cnt, "Time": datetime.now().strftime("%H:%M:%S"),
                    "ID": track_id, "Gender": attr['gender'], "Color": attr['color'],
                    "Direction": direction
                })

        # C. Draw
        tin, tout = ai_engine.get_counts()
        max_density = max(max_density, len(detections))
        fps_real = 1.0 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0

        # [UI 그리기] sv_line_zone을 넘겨줌
        frame = ui.draw_main_ui(frame, detections, labels, sv_line_zone, fps_real)
        frame = ui.draw_overlay_text(frame, tin+tout, len(detections), fps_real)

        display = ui.resize_for_display(frame)
        cv2.imshow("People Counter", display)

        proc_time = (time.time() - loop_start) * 1000
        wait = max(1, int(delay_ms - proc_time))
        if cv2.waitKey(wait) & 0xFF == ord('q'): break

    # End
    cap.release()
    cv2.destroyAllWindows()
    
    total_cnt = ai_engine.get_counts()[0] + ai_engine.get_counts()[1]
    
    final_report = ui.save_results(log_data, os.path.basename(source), 
                                   session_start, total_cnt, max_density)
    
    cv2.imshow("Final Report", final_report)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/dev_day.mp4')
    args = parser.parse_args()
    
    if os.path.exists(args.source):
        main(args.source)
    else:
        print("File not found.")