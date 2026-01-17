import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import *
from src.engine import VLMAttributeEngine
from src.utils import create_directories, save_best_samples, generate_report, show_summary_window

def main():
    create_directories()
    
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    import supervision as sv
    from ultralytics import YOLO

    print(f"[INFO] Loading YOLOv8 Detector from {MODEL_PATH}...")
    detector = YOLO(MODEL_PATH)
    vlm_engine = VLMAttributeEngine()
    
    target_video = DEFAULT_VIDEO_PATH 
    if not os.path.exists(target_video):
        print(f"[ERROR] Video file not found: {target_video}")
        return

    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream.")
        return

    # [FPS Control] Smart Sync + 30 FPS Limit
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = min(raw_fps, 30.0) if raw_fps > 0 else 30.0 
    frame_interval_ms = int(1000 / target_fps)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Target Video: {os.path.basename(target_video)}")
    print(f"[INFO] Speed Control: Native {raw_fps:.1f} FPS -> Sync to {target_fps:.1f} FPS")
    
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * target_fps))

    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    vlm_candidates = defaultdict(list)
    id_attributes = {} 
    best_crops_store = {}

    gender_stats = {"Male": 0, "Female": 0, "Unknown": 0}
    inference_log = []
    
    conf_scores_g = [] 
    conf_scores_c = []

    frame_idx = 0
    start_time = time.time()
    fusion_dist_limit = RESIZE_WIDTH * ID_FUSION_RATIO

    print(f"[INFO] Analysis Started...")

    while True:
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        fh, fw = int(frame.shape[0] * (RESIZE_WIDTH/frame.shape[1])), RESIZE_WIDTH
        frame = cv2.resize(frame, (fw, fh))
        display = frame.copy()
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        device = 0 if torch.cuda.is_available() else "cpu"
        results = detector(frame, verbose=False, device=device)[0]        
        detections = sv.Detections.from_ultralytics(results)
        
        valid_indices = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            aspect_ratio = bh / bw if bw > 0 else 0
            
            # 종횡비구분 1.4 -> 1.15
            if aspect_ratio > 1.15 and detections.confidence[i] >= CONF_THRESH:
                valid_indices.append(i)
        
        detections = detections[valid_indices]
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()

        if detections.tracker_id is not None:
            for xyxy, raw_id in zip(detections.xyxy, detections.tracker_id):
                raw_id = int(raw_id)
                track_hits[raw_id] += 1
                
                # 안정적인 트래킹을 위해 초기 몇 프레임은 스킵
                if track_hits[raw_id] < 3: continue 

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # [ID Mapping & Fusion Logic]
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

                # [VLM Analysis Trigger]
                if tid not in id_attributes:
                    # 3, 10, 20 프레임째에 이미지 수집
                    if track_hits[raw_id] in [3, 10, 20]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = vlm_engine.analyze(crop)
                            if res: vlm_candidates[tid].append({'res': res, 'img': crop.copy()})

                    # 31프레임째에 확정 
                    if track_hits[raw_id] == 31:
                        candidates = vlm_candidates[tid]
                        if candidates:
                            best_entry = max(candidates, key=lambda x: x['res']['g_cf'])
                            best_res = best_entry['res']
                            
                            final_g, final_c = best_res["g"], best_res["c"]
                            g_conf, c_conf = best_res["g_cf"], best_res["c_cf"]
                            vlm_desc = best_res.get("desc", "")

                            if g_conf < 50.0: final_g = "Unknown"
                            
                            id_attributes[tid] = (final_g, final_c, g_conf, c_conf)
                            gender_stats[final_g] += 1
                            
                            conf_scores_g.append(g_conf)
                            conf_scores_c.append(c_conf)
                            
                            log_str = f"[ID {tid:02d}] Result: {final_g}/{final_c} (G:{g_conf:.1f}% C:{c_conf:.1f}%) | {vlm_desc}"
                            inference_log.append(log_str)
                            
                            best_crops_store[tid] = {
                                'conf': g_conf,
                                'img': best_entry['img'],
                                'label': f"{final_g}_{final_c}"
                            }

                # [Display & Counting]
                attr = id_attributes.get(tid, ("Analyzing", "...", 0, 0))
                label = f"ID:{tid} | {attr[0][:1]}({attr[2]:.0f}%) | {attr[1]}({attr[3]:.0f}%)"

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    # 카운팅 기준 도달 시
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * target_fps):
                        counted_ids.add(tid)
                        total_count += 1

                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.8, color_box, 1)

        # [HUD Overlay]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"VLM ROBUST SYSTEM: {os.path.basename(target_video)}", (20, 45), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        
        elapsed_total = time.time() - start_time
        real_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0
        
        prog = int(frame_idx/total_frames*100) if total_frames > 0 else 0
        cv2.putText(display, f"PROG : {prog}% | FPS : {real_fps:.1f} (Limit: {target_fps:.0f})", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("AI Tracking System", display)
        
        proc_time_ms = (time.time() - loop_start_time) * 1000
        wait_ms = max(1, int(frame_interval_ms - proc_time_ms))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    
    # ==========================================
    # [Fix 3]  End-of-Stream Flush
    # 영상이 끝났을 때, 분석 중이던 ID들을 강제로 확정 짓고 카운팅에 포함시킴
    # ==========================================
    print(f"\n[INFO] End of Stream detected. Flushing remaining buffers...")
    
    # 1. 수집된 후보군(vlm_candidates)은 있으나, 아직 확정(id_attributes)되지 않은 ID 찾기
    for tid, candidates in vlm_candidates.items():
        if tid in id_attributes: 
            continue # 이미 처리된 ID는 패스

        if not candidates:
            continue # 후보 이미지가 하나도 없으면 패스

        # 가장 좋은 후보 선택 (로직 동일)
        best_entry = max(candidates, key=lambda x: x['res']['g_cf'])
        best_res = best_entry['res']
        
        final_g, final_c = best_res["g"], best_res["c"]
        g_conf, c_conf = best_res["g_cf"], best_res["c_cf"]
        vlm_desc = best_res.get("desc", "")

        if g_conf < 50.0: final_g = "Unknown"
        
        # 속성 확정
        id_attributes[tid] = (final_g, final_c, g_conf, c_conf)
        gender_stats[final_g] += 1
        
        conf_scores_g.append(g_conf)
        conf_scores_c.append(c_conf)
        
        # 로그에 [FLUSH] 태그를 달아서 영상 종료 후 처리되었음을 표시
        log_str = f"[ID {tid:02d}] Result: {final_g}/{final_c} (G:{g_conf:.1f}% C:{c_conf:.1f}%) | {vlm_desc} [FLUSHED]"
        inference_log.append(log_str)
        print(f"[FLUSH] Finalized ID {tid} at end of video.")

        best_crops_store[tid] = {
            'conf': g_conf,
            'img': best_entry['img'],
            'label': f"{final_g}_{final_c}"
        }

        # 2. 카운팅 누락 방지 (영상 끝부분에 등장해서 dwell time 부족했어도, 인식이 되었다면 카운트)
        # 이미 카운트 된 ID는 제외
        if tid not in counted_ids:
            counted_ids.add(tid)
            total_count += 1
            print(f"[FLUSH] Counted ID {tid} (Last minute appearance)")

    # ==========================================
    
    elapsed = time.time() - start_time
    final_fps = frame_idx/elapsed
    
    avg_g_conf = np.mean(conf_scores_g) if conf_scores_g else 0.0
    avg_c_conf = np.mean(conf_scores_c) if conf_scores_c else 0.0
    
    video_base_name = os.path.splitext(os.path.basename(target_video))[0]
    
    save_best_samples(video_base_name, best_crops_store)
    generate_report(video_base_name, total_count, gender_stats, avg_g_conf, avg_c_conf, final_fps, inference_log)
    show_summary_window(video_base_name, total_count, gender_stats, avg_g_conf, avg_c_conf, final_fps)

if __name__ == "__main__":
    main()