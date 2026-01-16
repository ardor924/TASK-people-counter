import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict
import torch

# src/ 폴더를 시스템 경로에 추가
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
    
    # 지연 로딩
    import supervision as sv
    from ultralytics import YOLO

    # 2. AI 엔진 로드
    print(f"[INFO] Loading YOLOv8 Detector from {MODEL_PATH}...")
    detector = YOLO(MODEL_PATH)
    vlm_engine = VLMAttributeEngine()
    
    # 영상 경로 설정
    target_video = DEFAULT_VIDEO_PATH 
    
    if not os.path.exists(target_video):
        print(f"[ERROR] Video file not found: {target_video}")
        return

    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream.")
        return

    # [FPS 제어 로직 - 스마트 싱크]
    # 영상의 원래 FPS를 가져오되, 너무 빠르면 30으로 제한 
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = min(raw_fps, 30.0) if raw_fps > 0 else 30.0
    frame_interval_ms = int(1000 / target_fps)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Target Video: {os.path.basename(target_video)}")
    print(f"[INFO] Speed Control: Native {raw_fps:.1f} FPS -> Sync to {target_fps:.1f} FPS")
    
    # 트래커 설정
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * target_fps))

    # 데이터 초기화
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

    frame_idx = 0
    start_time = time.time() # 전체 실행 시간 측정용
    fusion_dist_limit = RESIZE_WIDTH * ID_FUSION_RATIO

    print(f"[INFO] Analysis Started...")

    # 4. 메인 분석 루프
    while True:
        loop_start_time = time.time() # [Sync] 프레임 처리 시작 시간 측정

        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 프레임 리사이징
        fh, fw = int(frame.shape[0] * (RESIZE_WIDTH/frame.shape[1])), RESIZE_WIDTH
        frame = cv2.resize(frame, (fw, fh))
        display = frame.copy()
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        # 객체 탐지
        device = 0 if torch.cuda.is_available() else "cpu"
        results = detector(frame, verbose=False, device=device)[0]        
        detections = sv.Detections.from_ultralytics(results)
        
        # 필터링
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

                # ID Fusion
                if raw_id not in id_map:
                    found_legacy = False
                    for disp_id, data in list(track_registry.items()):
                        if disp_id in active_now_ids: continue 
                        if frame_idx - data["last_frame"] > (FORGET_TIME_SEC * target_fps):
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

                # VLM 분석
                if tid not in id_attributes:
                    if track_hits[raw_id] in [20, 30, 40]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = vlm_engine.analyze(crop)
                            if res: vlm_candidates[tid].append({'res': res, 'img': crop.copy()})

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

                # 카운팅
                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * target_fps):
                        counted_ids.add(tid)
                        total_count += 1

                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.8, color_box, 1)

        # UI 표시
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"VLM ROBUST SYSTEM: {os.path.basename(target_video)}", (20, 45), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        
        # [Sync] 실제 경과 시간 기준 FPS 표시
        elapsed_total = time.time() - start_time
        real_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0
        
        prog = int(frame_idx/total_frames*100) if total_frames > 0 else 0
        cv2.putText(display, f"PROG : {prog}% | FPS : {real_fps:.1f} (Limit: {target_fps:.0f})", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("AI Tracking System", display)
        
        # [Smart FPS Sync] 처리 시간을 고려하여 남은 시간만큼만 대기
        proc_time_ms = (time.time() - loop_start_time) * 1000
        wait_ms = max(1, int(frame_interval_ms - proc_time_ms))
        
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"): break

    # 5. 후처리
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    avg_conf = np.mean(conf_scores) if conf_scores else 0
    video_base_name = os.path.splitext(os.path.basename(target_video))[0]
    final_fps = frame_idx/elapsed
    
    save_best_samples(video_base_name, best_crops_store)
    generate_report(video_base_name, total_count, gender_stats, avg_conf, final_fps, inference_log)
    show_summary_window(video_base_name, total_count, gender_stats, avg_conf, final_fps)

if __name__ == "__main__":
    main()