import cv2
import argparse
import os
import time
import numpy as np
from datetime import datetime
from src.detector import PeopleDetector
from src.attributes import AttributeExtractor
from src.counter import PeopleCounter
from src.database import DatabaseManager

def generate_report_image(video_name, start_time, end_time, loop_count, last_count, max_density, loop_results):
    """
    최종 결과 리포트 이미지 생성
    """
    h, w = 500, 600
    report_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    gray = (200, 200, 200)
    cyan = (255, 255, 0)
    
    # 제목
    cv2.putText(report_img, "Monitoring Session Report", (120, 50), font, 1, white, 2)
    cv2.line(report_img, (50, 70), (550, 70), white, 2)
    
    # 기본 정보
    cv2.putText(report_img, f"Target: {video_name}", (50, 120), font, 0.6, white, 1)
    cv2.putText(report_img, f"Session Start: {start_time}", (50, 150), font, 0.6, white, 1)
    cv2.putText(report_img, f"Session End:   {end_time}", (50, 180), font, 0.6, white, 1)
    
    cv2.line(report_img, (50, 210), (550, 210), gray, 1)

    # 통계
    cv2.putText(report_img, f"Total Loops: {loop_count}", (50, 250), font, 0.8, white, 1)
    
    # 마지막 결과
    cv2.putText(report_img, "Latest Result:", (50, 300), font, 0.8, yellow, 1)
    cv2.putText(report_img, f"- Count: {last_count} People", (70, 340), font, 1, green, 2)
    cv2.putText(report_img, f"- Max Density: {max_density}", (70, 380), font, 0.8, white, 1)
    
    # History
    cv2.line(report_img, (50, 410), (550, 410), gray, 1)
    cv2.putText(report_img, "[ History ]", (50, 440), font, 0.6, cyan, 1)
    
    history_text = ""
    if not loop_results:
        history_text = "Single Run Completed."
    else:
        items = [f"L{i+1}:{cnt}" for i, cnt in enumerate(loop_results)]
        history_text = " | ".join(items[-5:]) 
        if len(loop_results) > 5:
            history_text = "... " + history_text
            
    cv2.putText(report_img, history_text, (150, 440), font, 0.6, white, 1)

    # 파일 저장
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    clean_name = os.path.splitext(video_name)[0]
    file_timestamp = datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")
    filename = f"Report_{clean_name}_{file_timestamp}.jpg"
    save_path = os.path.join(save_dir, filename)
    
    cv2.imwrite(save_path, report_img)
    print(f"📄 Report saved to: {save_path}")
    
    return report_img

def main(video_path, is_loop_mode):
    # 1. 초기화
    model_name = 'yolov8n.pt'
    detector = PeopleDetector(model_name=model_name) 
    attr_extractor = AttributeExtractor()
    counter = PeopleCounter(line_height_ratio=0.55)
    
    db = DatabaseManager()
    print(f"DB initialized at: {db.db_path}")

    session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_name = os.path.basename(video_path)
    
    max_density_record = 0
    loop_count = 1
    loop_results = [] 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 영상을 열 수 없습니다 -> {video_path}")
        return

    track_info = {} 
    prev_time = 0
    frame_count = 0

    print(f"Processing: {video_path}")
    print(f"Mode: {'🔄 Loop Mode' if is_loop_mode else '▶️ Single Run Mode'}")
    print("종료하려면 'q' 키를 누르세요.")
    
    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        
        # --- [영상 종료 시점 처리] ---
        if not ret:
            print(f"\n✅ Processing Finished (Loop {loop_count})")
            print(f"   => Result: {counter.count} People Detected.")
            
            # 결과 저장
            loop_results.append(counter.count)
            loop_video_name = f"{video_name} (Loop {loop_count})"
            db.save_summary(loop_video_name, session_start_time, counter.count)
            
            if is_loop_mode:
                print("🔄 Restarting video (Loop Mode)...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                counter.count = 0
                counter.counted_ids.clear()
                track_info.clear()
                max_density_record = 0
                loop_count += 1
                
                del detector
                detector = PeopleDetector(model_name=model_name)
                continue
            else:
                break
        # ---------------------------
        
        frame_count += 1
        h, w = frame.shape[:2]

        # 추적
        result = detector.track(frame)
        
        current_ids = []
        if result.boxes is not None and result.boxes.id is not None:
            current_ids = result.boxes.id.cpu().numpy().astype(int)

        # 데이터 처리
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, current_ids):
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                center_y = int((y1 + y2) / 2)

                # 카운팅 & DB
                if counter.update(center_y, h, track_id):
                    info = track_info.get(track_id, {"color": "Unknown", "gender": "Unknown"})
                    db.insert_log(track_id, info['gender'], info['color'])

                # 속성 분석
                if track_id not in track_info or frame_count % 30 == 0:
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.shape[0] > 10 and person_img.shape[1] > 10:
                        color = attr_extractor.extract_color(person_img)
                        gender = attr_extractor.extract_gender(person_img)
                        track_info[track_id] = {"color": color, "gender": gender}
                    else:
                        track_info[track_id] = {"color": "Unknown", "gender": "Unknown"}
                
                info = track_info.get(track_id, {"color": "Unknown", "gender": "Unknown"})
                
                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} | {info['gender']} | {info['color']}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + lw + 10, y1), (0, 255, 0), -1) 
                cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # UI
        counter.draw(frame)

        # Density
        current_density = len(current_ids)
        if current_density > max_density_record:
            max_density_record = current_density
            
        cv2.rectangle(frame, (15, 80), (280, 125), (0, 0, 0), -1)
        cv2.putText(frame, f"Density: {current_density} Person", (30, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # FPS
        delta_time = time.time() - prev_time
        fps = 1 / delta_time if delta_time > 0 else 0
        prev_time = curr_time
        
        fps_text = f"FPS: {int(fps)}"
        (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (w - fw - 30, 15), (w - 10, 60), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (w - fw - 20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if is_loop_mode:
            cv2.putText(frame, f"Loop: {loop_count}", (w - 150, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("ReID People Counter", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
# --- 종료 처리 ---
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not is_loop_mode and counter.count > 0:
         db.save_summary(video_name, session_start_time, counter.count)
    
    # 터미널에 DB 내용 일부 출력 (검증용)
    db.print_recent_logs()
    
    # [수정] CSV 내보내기 시 video_name 전달
    db.export_to_csv(video_name)
    
    db.close()
    
    report_img = generate_report_image(
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
    parser.add_argument('--source', type=str, default='data/dev_day.mp4', help='Video file path')
    parser.add_argument('--loop', action='store_true', help='Enable infinite loop mode')
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: 파일을 찾을 수 없습니다 -> {args.source}")
    else:
        main(args.source, args.loop)