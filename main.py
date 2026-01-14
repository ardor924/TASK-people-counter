import cv2
import argparse
import os
import time
import threading
import numpy as np
from datetime import datetime

# --- [모듈화된 클래스 Import] ---
from src.voting import VotingProcessor
from src.report import ReportGenerator

# 전역 로딩 상태 관리
loading_status = {
    "is_loaded": False,
    "detector": None,
    "attr_extractor": None,
    "counter": None,
    "modules": None
}

def load_ai_models(line_height_ratio):
    """ 백그라운드 모델 로딩 (DB 제외) """
    try:
        from src.detector import PeopleDetector
        from src.attributes import AttributeExtractor
        from src.counter import PeopleCounter
        from src.database import DatabaseManager
        
        detector = PeopleDetector(model_name='yolov8n.pt')
        attr_extractor = AttributeExtractor()
        counter = PeopleCounter(line_height_ratio=line_height_ratio)
        
        loading_status["detector"] = detector
        loading_status["attr_extractor"] = attr_extractor
        loading_status["counter"] = counter
        loading_status["modules"] = {
            "PeopleDetector": PeopleDetector,
            "AttributeExtractor": AttributeExtractor,
            "PeopleCounter": PeopleCounter,
            "DatabaseManager": DatabaseManager
        }
        loading_status["is_loaded"] = True
    except Exception as e:
        print(f"\n❌ Critical Error during loading: {e}")
        os._exit(1)

def main(video_path, is_loop_mode):
    # 1. 영상 열기 및 초기화
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 영상을 열 수 없습니다 -> {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret: return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 2. 로딩 스레드 시작
    print("⏳ UI Started. Loading models...")
    threading.Thread(target=load_ai_models, args=(0.55,), daemon=True).start()

    # --- [로딩 화면 UI] ---
    dots = ""
    last_dot_time = time.time()
    screen_w, screen_h = 960, 540

    while not loading_status["is_loaded"]:
        if time.time() - last_dot_time > 0.5:
            dots += "."
            if len(dots) > 3: dots = ""
            last_dot_time = time.time()
        
        loading_frame = cv2.resize(first_frame.copy(), (screen_w, screen_h))
        loading_frame = cv2.addWeighted(loading_frame, 0.4, np.zeros(loading_frame.shape, loading_frame.dtype), 0, 0)

        text_main = f"System Loading{dots}"
        text_sub = "Initializing PyTorch & YOLOv8..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (tw_m, th_m), _ = cv2.getTextSize(text_main, font, 1.2, 2)
        (tw_s, th_s), _ = cv2.getTextSize(text_sub, font, 0.6, 1)
        
        cv2.putText(loading_frame, text_main, ((screen_w - tw_m)//2, (screen_h + th_m)//2), font, 1.2, (0, 255, 255), 2)
        cv2.putText(loading_frame, text_sub, ((screen_w - tw_s)//2, (screen_h + th_m)//2 + 40), font, 0.6, (200, 200, 200), 1)

        cv2.imshow("ReID People Counter", loading_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): return

    # --- [시스템 준비 완료] ---
    detector = loading_status["detector"]
    attr_extractor = loading_status["attr_extractor"]
    counter = loading_status["counter"]
    DatabaseManagerClass = loading_status["modules"]["DatabaseManager"]
    PeopleDetectorClass = loading_status["modules"]["PeopleDetector"]
    
    db = DatabaseManagerClass()
    # [수정] max_history를 30으로 늘려서 더 안정적으로 (1초 정도의 기록 유지)
    voting_processor = VotingProcessor(max_history=30) 
    report_generator = ReportGenerator()               

    print(f"✅ System Loaded! DB initialized.")
    
    session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_name = os.path.basename(video_path)
    
    final_attributes = {} # 화면 표시용
    
    max_density_record = 0
    loop_count = 1
    loop_results = []
    
    frame_count = 0
    prev_time = 0
    paused = False 

    print(f"Processing: {video_path}")
    print(f"Mode: {'🔄 Loop Mode' if is_loop_mode else '▶️ Single Run Mode'}")

    # --- [메인 루프] ---
    while True:
        curr_time = time.time()
        
        # Pause Check
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): 
            paused = not paused
            print("⏸ Paused" if paused else "▶ Resumed")
        
        if paused: continue

        ret, frame = cap.read()
        
        # Loop Check
        if not ret:
            print(f"\n✅ Loop {loop_count} Finished. Count: {counter.count}")
            loop_results.append(counter.count)
            db.save_summary(f"{video_name} (Loop {loop_count})", session_start_time, counter.count)
            
            if is_loop_mode:
                print("🔄 Restarting video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                counter.count = 0
                counter.counted_ids.clear()
                voting_processor.clear() 
                final_attributes.clear()
                max_density_record = 0
                loop_count += 1
                del detector
                detector = PeopleDetectorClass(model_name='yolov8n.pt')
                continue
            else:
                break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # 3. Detection & Tracking
        result = detector.track(frame)
        
        current_ids = []
        if result.boxes is not None and result.boxes.id is not None:
            current_ids = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, current_ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = box
                center_y = int((y1 + y2) / 2)

                # 4. Attribute Analysis & Voting (3프레임 주기로 빈도 상향)
                if frame_count % 3 == 0:
                    person_img = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if person_img.size > 0:
                        c_val, c_conf = attr_extractor.extract_color(person_img)
                        g_val, g_conf = attr_extractor.extract_gender(person_img)
                        
                        # [Voting 업데이트]
                        voting_processor.update(track_id, g_val, g_conf, c_val, c_conf)

                # 5. 결과 가져오기 (매 프레임)
                final_attributes[track_id] = voting_processor.get_result(track_id)
                info = final_attributes[track_id]

                # 6. Counting & DB
                if counter.update(center_y, h, track_id):
                    db.insert_log(track_id, info['gender'], info['gender_conf'], 
                                  info['color'], info['color_conf'])

                # 7. Visualization (Box & Label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                g_str = info['gender']
                g_disp = f"{g_str[0]}({int(info['gender_conf']*100)}%)" if g_str in ["Male", "Female"] else "Unk"
                c_conf_disp = int(info['color_conf']*100)
                label = f"ID:{track_id}|{g_disp}|{info['color']}({c_conf_disp}%)"
                
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + lw + 10, y1), (0, 255, 0), -1) 
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 8. UI Overlay (FPS, Density, etc.)
        counter.draw(frame)

        current_density = len(current_ids)
        if current_density > max_density_record:
            max_density_record = current_density
            
        cv2.rectangle(frame, (15, 80), (280, 125), (0, 0, 0), -1)
        cv2.putText(frame, f"Density: {current_density} Person", (30, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        delta_time = time.time() - prev_time
        fps = 1 / delta_time if delta_time > 0 else 0
        prev_time = curr_time
        
        fps_text = f"FPS: {int(fps)}"
        (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (w - fw - 30, 15), (w - 10, 60), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (w - fw - 20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if is_loop_mode:
            loop_text = f"Loop: {loop_count}"
            (lw, lh), _ = cv2.getTextSize(loop_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, loop_text, (w - lw - 20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("ReID People Counter", cv2.resize(frame, (screen_w, screen_h)))

    # --- [종료 처리] ---
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not is_loop_mode and counter.count > 0:
         db.save_summary(video_name, session_start_time, counter.count)
    
    db.print_recent_logs()
    db.export_to_csv(video_name)
    db.close()
    
    report_img = report_generator.generate_and_save(
        video_name, session_start_time, end_time_str, 
        loop_count, counter.count, max_density_record, loop_results
    )
    
    cv2.imshow("Session Report", report_img)
    print("결과 리포트를 화면에 표시합니다. (아무 키나 누르면 종료)")
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/dev_day.mp4')
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    if os.path.exists(args.source): main(args.source, args.loop)