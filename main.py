import cv2
import argparse
import os
import time
import threading
import numpy as np
from datetime import datetime

# --- [Custom Modules] ---
from src.detector import PeopleDetector
from src.attributes import AttributeExtractor
from src.counter import PeopleCounter
from src.database import DatabaseManager
from src.voting import VotingProcessor
from src.report import ReportGenerator

loading_status = {"is_loaded": False}
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# [키 입력 관리] 전역 변수
paused = False
exit_flag = False

def load_ai_models(line_height_ratio):
    try:
        global detector, attr_extractor, counter, voting_processor, report_generator
        
        model_name = os.path.join("models", "yolov8n.pt")
        if not os.path.exists(model_name): model_name = 'yolov8n.pt'
        
        detector = PeopleDetector(model_name=model_name)
        attr_extractor = AttributeExtractor() 
        counter = PeopleCounter(line_height_ratio=line_height_ratio)
        voting_processor = VotingProcessor(max_history=30)
        report_generator = ReportGenerator()
        
        loading_status["is_loaded"] = True
    except Exception as e:
        print(f"\n❌ Load Error: {e}")
        os._exit(1)

def draw_info_box(img, text, x, y, bg_color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (t_w, t_h), base = cv2.getTextSize(text, font, scale, thickness)
    # 텍스트 배경 (검정색 테두리 효과 + 배경)
    cv2.rectangle(img, (x, y - t_h - 6), (x + t_w + 4, y + base + 2), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, scale, (0, 0, 0), thickness)

def check_key_input(wait_time=1):
    """ [반응 속도 개선] 키 입력을 루프 중간중간 확인 """
    global paused, exit_flag
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord(' '):
        paused = not paused
    elif key == ord('q'):
        exit_flag = True

def main(video_path, is_loop_mode):
    global paused, exit_flag
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {video_path}")
        return

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    if fps_val == 0: fps_val = 30
    delay = int(1000 / fps_val)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. 즉시 로딩 화면
    ret, first_frame = cap.read()
    if not ret: return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    loading_screen = cv2.resize(first_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    loading_screen = cv2.addWeighted(loading_screen, 0.3, np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8), 0, 0)
    cv2.putText(loading_screen, "Now Loading...", (DISPLAY_WIDTH//2 - 120, DISPLAY_HEIGHT//2), font, 1.2, (0, 255, 255), 2)
    cv2.imshow("People Counter", loading_screen)
    cv2.waitKey(1) 

    # 2. 모델 로딩
    print("⏳ Loading AI Models...")
    threading.Thread(target=load_ai_models, args=(0.55,), daemon=True).start()

    dots = ""
    last_tick = time.time()
    while not loading_status["is_loaded"]:
        if time.time() - last_tick > 0.5:
            dots = "." * ((len(dots) + 1) % 4)
            last_tick = time.time()
            temp = loading_screen.copy()
            cv2.putText(temp, f"Now Loading{dots}", (DISPLAY_WIDTH//2 - 120, DISPLAY_HEIGHT//2), font, 1.2, (0, 255, 255), 2)
            cv2.imshow("People Counter", temp)
            if cv2.waitKey(100) & 0xFF == ord('q'): return

    # 3. 메인 로직 준비
    db = DatabaseManager()
    print("✅ System Ready.")

    session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_name = os.path.basename(video_path)
    
    analysis_cache = {} 
    loop_cnt = 1
    max_density = 0
    frame_count = 0
    loop_results = []
    
    while True:
        # [키 입력 확인 1] 루프 시작 시점
        check_key_input(1)
        if exit_flag: break

        if paused:
            pause_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.putText(pause_frame, "PAUSED", (DISPLAY_WIDTH//2 - 80, DISPLAY_HEIGHT//2), font, 1.5, (0, 0, 255), 3)
            cv2.imshow("People Counter", pause_frame)
            cv2.waitKey(30)
            continue

        loop_start_time = time.time()
        ret, frame = cap.read()
        
        # 영상 종료 처리
        if not ret:
            print(f"✅ Loop {loop_cnt} End. Count: {counter.count}")
            loop_results.append(counter.count)
            db.save_summary(f"{video_name} (Loop {loop_cnt})", session_start, counter.count)
            
            if is_loop_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                counter.count = 0
                counter.counted_ids.clear()
                loop_cnt += 1
                continue
            else:
                break

        frame_count += 1
        h, w = frame.shape[:2]

        # [키 입력 확인 2] 무거운 AI 돌리기 전 한 번 더 체크
        check_key_input(1)
        if paused: continue

        # 1. Tracking
        result = detector.track(frame)
        
        current_ids = []
        boxes = []
        confs = []

        if result.boxes is not None and result.boxes.id is not None:
            current_ids = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

        valid_cnt = 0
        
        for id, box, conf in zip(current_ids, boxes, confs):
            # 카운팅 기준 (ID가 튀어도 라인은 놓치지 않게)
            if conf < 0.35: continue
            valid_cnt += 1

            x1, y1, x2, y2 = box
            cy = (y1 + y2) // 2
            
            # 2. 속성 분석 (Load Balancing)
            is_center = (h * 0.4 < cy < h * 0.6)
            last_frame = analysis_cache.get(id, -999)
            
            if id not in analysis_cache or (is_center and (frame_count - last_frame > 10)):
                person_img = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if person_img.size > 0:
                    g, g_c, c, c_c = attr_extractor.extract_attributes(person_img)
                    voting_processor.update(id, g, g_c, c, c_c)
                    analysis_cache[id] = frame_count

            # 3. 정보 가져오기
            info = voting_processor.get_result(id)
            
            # 4. 카운팅
            if counter.update(cy, h, id):
                db.insert_log(id, info['gender'], info['gender_conf'], 
                              info['color'], info['color_conf'])

            # 5. 시각화
            color = (0, 255, 0)
            if id > 100: color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # [요청 양식 완벽 적용] ID:1 | M (80%) | Blue (70%)
            g_str = "M" if info['gender'] == "Male" else "F" if info['gender'] == "Female" else "?"
            g_conf_str = f"({int(info['gender_conf']*100)}%)"
            
            c_str = info['color']
            c_conf_str = f"({int(info['color_conf']*100)}%)"
            
            label = f"ID:{id} | {g_str} {g_conf_str} | {c_str} {c_conf_str}"
            draw_info_box(frame, label, x1, y1 - 5, bg_color=color)

        max_density = max(max_density, valid_cnt)

        # UI Overlay (원본 프레임에 그림)
        counter.draw(frame)
        
        # [Density 박스]
        cv2.rectangle(frame, (20, 90), (280, 135), (0, 0, 0), -1)
        cv2.putText(frame, f"Density: {valid_cnt}", (35, 125), font, 0.8, (0, 255, 0), 2)
        
        # FPS
        elapsed = (time.time() - loop_start_time) * 1000
        wait_ms = max(1, delay - int(elapsed))
        curr_fps = 1000 / (elapsed + wait_ms) if (elapsed + wait_ms) > 0 else 0
        
        fps_label = f"FPS: {int(curr_fps)}"
        (fw, fh), _ = cv2.getTextSize(fps_label, font, 1, 2)
        cv2.rectangle(frame, (w - fw - 30, 20), (w - 10, 60), (0, 0, 0), -1)
        cv2.putText(frame, fps_label, (w - fw - 20, 50), font, 1, (0, 255, 255), 2)

        if is_loop_mode:
            cv2.putText(frame, f"Loop: {loop_cnt}", (w - 150, 90), font, 0.7, (200, 200, 200), 2)

        # 화면 출력
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("People Counter", display_frame)
        
        # [키 입력 확인 3] 렌더링 후 최종 딜레이 (FPS 유지)
        check_key_input(wait_ms)
        if exit_flag: break

    # 종료 처리
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not is_loop_mode and counter.count > 0:
         db.save_summary(video_name, session_start, counter.count)

    db.export_to_csv(video_name)
    db.close()
    
    report = report_generator.generate_and_save(
        video_name, session_start, end_time, 
        loop_cnt, counter.count, max_density, loop_results
    )
    cv2.imshow("Final Report", report)
    print("✅ Process Finished. Press any key to exit.")
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/dev_day.mp4')
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    if os.path.exists(args.source): main(args.source, args.loop)