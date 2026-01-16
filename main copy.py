import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict
import torch

# src/ 폴더를 시스템 경로에 추가 (모듈 참조 안전성 확보)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 모듈 임포트
from src.config import *
from src.engine import VLMAttributeEngine
from src.utils import create_directories, save_best_samples, generate_report, show_summary_window

def main():
    # 1. 초기화 및 필수 폴더 생성
    create_directories()
    
    # 윈도우 설정
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    # 지연 로딩 (YOLO, Supervision)
    import supervision as sv
    from ultralytics import YOLO

    # 2. AI 엔진 및 탐지기 로드
    print(f"[INFO] Loading YOLOv8 Detector from {MODEL_PATH}...")
    detector = YOLO(MODEL_PATH)
    vlm_engine = VLMAttributeEngine()
    
    # [수정됨] 영상 경로 설정 (config.py의 변수 사용)
    # VIDEO_PATH 변수가 아니라 config.py에서 가져온 DEFAULT_VIDEO_PATH를 사용합니다.
    target_video = DEFAULT_VIDEO_PATH 
    
    if not os.path.exists(target_video):
        print(f"[ERROR] Video file not found: {target_video}")
        print("Please check 'src/config.py' or 'data/' folder.")
        return

    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream.")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wait_time = max(1, int(1000 / src_fps))
    
    # 트래커 설정
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * src_fps))

    # 3. 데이터 및 통계 변수 초기화
    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    vlm_candidates = defaultdict(list)
    id_attributes = {} 
    best_crops_store = {}

    gender_stats = {"Male": 0, "Female": 0, "Unknown": 0}
    inference_log = []
    conf_scores = [] 

    frame_idx, start_time = 0, time.time()
    fusion_dist_limit = RESIZE_WIDTH * ID_FUSION_RATIO

    print(f"[INFO] Analysis Started: {os.path.basename(target_video)}")

    # 4. 메인 분석 루프
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 프레임 리사이징 및 ROI 표시
        fh, fw = int(frame.shape[0] * (RESIZE_WIDTH/frame.shape[1])), RESIZE_WIDTH
        frame = cv2.resize(frame, (fw, fh))
        display = frame.copy()
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        # 객체 탐지 (YOLO)
        # GPU 사용 가능 여부 체크
        device = 0 if torch.cuda.is_available() else "cpu"
        results = detector(frame, verbose=False, device=device)[0]        
        
        detections = sv.Detections.from_ultralytics(results)
        
        # 기하학적 필터링 (Aspect Ratio)
        valid_indices = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            aspect_ratio = bh / bw if bw > 0 else 0
            if aspect_ratio > 1.5 and detections.confidence[i] >= CONF_THRESH:
                valid_indices.append(i)
        
        detections = detections[valid_indices]
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()

        if detections.tracker_id is not None:
            for xyxy, raw_id in zip(detections.xyxy, detections.tracker_id):
                raw_id = int(raw_id)
                track_hits[raw_id] += 1
                if track_hits[raw_id] < 10: continue 

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ID Fusion 알고리즘
                if raw_id not in id_map:
                    found_legacy = False
                    for disp_id, data in list(track_registry.items()):
                        if disp_id in active_now_ids: continue 
                        if frame_idx - data["last_frame"] > (FORGET_TIME_SEC * src_fps):
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

                # VLM 분석 (Sparse Sampling: 20, 30, 40 프레임)
                if tid not in id_attributes:
                    if track_hits[raw_id] in [20, 30, 40]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = vlm_engine.analyze(crop)
                            if res: vlm_candidates[tid].append({'res': res, 'img': crop.copy()})

                    # 분석 확정 및 Best Frame 채택
                    if track_hits[raw_id] == 41:
                        candidates = vlm_candidates[tid]
                        if candidates:
                            best_entry = max(candidates, key=lambda x: x['res']['g_cf'])
                            best_res = best_entry['res']
                            
                            final_g, final_c = best_res["g"], best_res["c"]
                            g_conf, c_conf = best_res["g_cf"], best_res["c_cf"]
                            if g_conf < 50.0: final_g = "Unknown"
                            
                            id_attributes[tid] = (final_g, final_c, g_conf, c_conf)
                            gender_stats[final_g] += 1
                            conf_scores.append(g_conf)
                            
                            inference_log.append(f"[ID {tid:02d}] Best G: {g_conf:4.1f}% | C: {c_conf:4.1f}% | Res: {final_g}/{final_c}")
                            
                            best_crops_store[tid] = {
                                'conf': g_conf,
                                'img': best_entry['img'],
                                'label': f"{final_g}_{final_c}"
                            }

                attr = id_attributes.get(tid, ("Analyzing", "...", 0, 0))
                label = f"ID:{tid} | {attr[0][:1]}({attr[2]:.0f}%) | {attr[1]}({attr[3]:.0f}%)"

                # 카운팅 로직 (ROI 통과 시)
                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * src_fps):
                        counted_ids.add(tid)
                        total_count += 1

                # 바운딩 박스 및 텍스트 렌더링
                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.8, color_box, 1)

        # 상단 UI 패널
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"VLM ROBUST SYSTEM: {os.path.basename(target_video)}", (20, 45), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        
        prog = int(frame_idx/total_frames*100) if total_frames > 0 else 0
        cv2.putText(display, f"PROGRESS : {prog}% | FPS : {frame_idx/(time.time()-start_time):.1f}", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("AI Tracking System", display)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"): break

    # 5. 후처리 및 리포트 저장
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    avg_conf = np.mean(conf_scores) if conf_scores else 0
    video_base_name = os.path.splitext(os.path.basename(target_video))[0]
    final_fps = frame_idx/elapsed
    
    # 유틸리티 함수 호출
    save_best_samples(video_base_name, best_crops_store)
    generate_report(video_base_name, total_count, gender_stats, avg_conf, final_fps, inference_log)
    show_summary_window(video_base_name, total_count, gender_stats, avg_conf, final_fps)

if __name__ == "__main__":
    main()