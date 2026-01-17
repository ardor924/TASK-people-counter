import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict
import torch

# 프로젝트 모듈 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import *
from src.engine import VLMAttributeEngine
from src.utils import create_directories, save_best_samples, generate_report, show_summary_window

def main():
    # ==========================================
    # 1. 초기 환경 설정 및 모델 로드
    # ==========================================
    create_directories() # 로그 및 샘플 저장용 디렉토리 생성
    
    cv2.namedWindow("AI Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Tracking System", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))
    
    import supervision as sv
    from ultralytics import YOLO

    print(f"[INFO] Loading YOLOv8 Detector from {MODEL_PATH}...")
    detector = YOLO(MODEL_PATH) # 객체 탐지 모델 로드
    vlm_engine = VLMAttributeEngine() # VLM(CLIP) 속성 분석 엔진 초기화
    
    target_video = DEFAULT_VIDEO_PATH 
    if not os.path.exists(target_video):
        print(f"[ERROR] Video file not found: {target_video}")
        return

    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream.")
        return

    # [FPS Control] 원본 영상 속도와 처리 속도를 동기화하기 위한 설정
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = min(raw_fps, 30.0) if raw_fps > 0 else 30.0 
    frame_interval_ms = int(1000 / target_fps)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Target Video: {os.path.basename(target_video)}")
    print(f"[INFO] Speed Control: Native {raw_fps:.1f} FPS -> Sync to {target_fps:.1f} FPS")
    
    # [Tracker] ByteTrack 초기화 (유실된 트랙을 보관하는 버퍼 설정 포함)
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * target_fps))

    # 데이터 관리용 변수들
    track_hits = defaultdict(int) # 객체별 등장 프레임 수
    id_map, next_display_id = {}, 1 # Re-ID 및 가독성을 위한 ID 매핑
    track_registry = {} # ID Fusion을 위한 위치 정보 저장소
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0 # 카운팅 로직
    
    vlm_candidates = defaultdict(list) # 분석 후보 프레임 보관용
    id_attributes = {} # 확정된 객체 속성
    best_crops_store = {} # 신뢰도 상위 샘플 보관

    gender_stats = {"Male": 0, "Female": 0, "Unknown": 0}
    inference_log = [] # 최종 리포트용 로그
    conf_scores_g, conf_scores_c = [], [] # 평균 신뢰도 산출용

    frame_idx = 0
    start_time = time.time()
    fusion_dist_limit = RESIZE_WIDTH * ID_FUSION_RATIO # ID 통합을 위한 거리 임계값

    print(f"[INFO] Analysis Started...")

    # ==========================================
    # 2. 메인 처리 루프 (Main Inference Loop)
    # ==========================================
    while True:
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 성능과 정확도의 균형을 위한 리사이징
        fh, fw = int(frame.shape[0] * (RESIZE_WIDTH/frame.shape[1])), RESIZE_WIDTH
        frame = cv2.resize(frame, (fw, fh))
        display = frame.copy()
        
        # ROI 카운팅 라인 설정
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        # [Detection] YOLOv8 객체 탐지 수행
        device = 0 if torch.cuda.is_available() else "cpu"
        results = detector(frame, verbose=False, device=device)[0]        
        detections = sv.Detections.from_ultralytics(results)
        
        # [Filtering] 객체 유효성 검증 (종횡비 필터링을 통해 노이즈 제거)
        valid_indices = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            aspect_ratio = bh / bw if bw > 0 else 0
            
            # 사람이 걷는 자세를 고려한 종횡비(1.15) 및 신뢰도 임계값 적용
            if aspect_ratio > 1.15 and detections.confidence[i] >= CONF_THRESH:
                valid_indices.append(i)
        
        detections = detections[valid_indices]
        # [Tracking] ByteTrack을 통한 실시간 추적 갱신
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()

        if detections.tracker_id is not None:
            for xyxy, raw_id in zip(detections.xyxy, detections.tracker_id):
                raw_id = int(raw_id)
                track_hits[raw_id] += 1
                
                # 안정적인 트래킹이 확보된 프레임(3회 이상)부터 처리 시작
                if track_hits[raw_id] < 3: continue 

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ==========================================
                # 3. ID Fusion & Re-ID Logic (ID 중복 생성 방지)
                # ==========================================
                if raw_id not in id_map:
                    found_legacy = False
                    for disp_id, data in list(track_registry.items()):
                        if disp_id in active_now_ids: continue # 현재 활성화된 객체는 스킵
                        
                        # 일정 시간(FORGET_TIME) 이내에 사라진 객체만 재매칭 시도
                        if frame_idx - data["last_frame"] > (FORGET_TIME_SEC * target_fps):
                            if disp_id in track_registry: del track_registry[disp_id]
                            continue
                        
                        # 유클리드 거리 기반 근접도 측정 (Re-ID)
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

                # ==========================================
                # 4. VLM Attribute Analysis (다단계 시계열 투표)
                # ==========================================
                if tid not in id_attributes:
                    # 연산 부하를 줄이기 위해 3, 10, 20 프레임 시점에서만 분석 수행
                    if track_hits[raw_id] in [3, 10, 20]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = vlm_engine.analyze(crop) # VLM/HSV 하이브리드 추론
                            if res: vlm_candidates[tid].append({'res': res, 'img': crop.copy()})

                    # 31프레임 시점에서 수집된 후보군 중 가장 신뢰도가 높은 결과를 최종 확정
                    if track_hits[raw_id] == 31:
                        candidates = vlm_candidates[tid]
                        if candidates:
                            # 신뢰도(Confidence) 기반 최적 프레임 선택
                            best_entry = max(candidates, key=lambda x: x['res']['g_cf'])
                            best_res = best_entry['res']
                            
                            final_g, final_c = best_res["g"], best_res["c"]
                            g_conf, c_conf = best_res["g_cf"], best_res["c_cf"]
                            vlm_desc = best_res.get("desc", "")

                            if g_conf < 50.0: final_g = "Unknown" # 저신뢰도 데이터 필터링
                            
                            id_attributes[tid] = (final_g, final_c, g_conf, c_conf)
                            gender_stats[final_g] += 1
                            
                            conf_scores_g.append(g_conf)
                            conf_scores_c.append(c_conf)
                            
                            log_str = f"[ID {tid:02d}] Result: {final_g}/{final_c} (G:{g_conf:.1f}% C:{c_conf:.1f}%) | {vlm_desc}"
                            inference_log.append(log_str)
                            
                            # 우수성 입증용 상위 샘플 저장소에 추가
                            best_crops_store[tid] = {
                                'conf': g_conf,
                                'img': best_entry['img'],
                                'label': f"{final_g}_{final_c}"
                            }

                # ==========================================
                # 5. ROI 카운팅 및 화면 출력 (Dwell-Time 기반)
                # ==========================================
                attr = id_attributes.get(tid, ("Analyzing", "...", 0, 0))
                label = f"ID:{tid} | {attr[0][:1]}({attr[2]:.0f}%) | {attr[1]}({attr[3]:.0f}%)"

                # 판정선(ROI) 아래에 존재하며 일정 시간(Dwell Time) 유지 시 카운트
                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * target_fps):
                        counted_ids.add(tid)
                        total_count += 1

                # 시각화 (카운트된 객체는 녹색 박스 표시)
                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.8, color_box, 1)

        # [HUD Overlay] 실시간 시스템 상태 정보 출력
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
        
        # FPS 제어 및 루프 탈출 조건
        proc_time_ms = (time.time() - loop_start_time) * 1000
        wait_ms = max(1, int(frame_interval_ms - proc_time_ms))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    
    # ==========================================
    # 6. 유종의 미 (End-of-Stream Flush 로직)
    # ==========================================
    # 영상 종료 시 분석 중이던 잔여 버퍼 데이터를 강제로 확정 및 카운트 반영
    print(f"\n[INFO] End of Stream detected. Flushing remaining buffers...")
    
    for tid, candidates in vlm_candidates.items():
        if tid in id_attributes or not candidates: continue

        best_entry = max(candidates, key=lambda x: x['res']['g_cf'])
        best_res = best_entry['res']
        final_g, final_c = best_res["g"], best_res["c"]
        g_conf, c_conf = best_res["g_cf"], best_res["c_cf"]

        if g_conf < 50.0: final_g = "Unknown"
        
        id_attributes[tid] = (final_g, final_c, g_conf, c_conf)
        gender_stats[final_g] += 1
        conf_scores_g.append(g_conf)
        conf_scores_c.append(c_conf)
        
        inference_log.append(f"[ID {tid:02d}] Result: {final_g}/{final_c} (G:{g_conf:.1f}% C:{c_conf:.1f}%) [FLUSHED]")
        best_crops_store[tid] = {'conf': g_conf, 'img': best_entry['img'], 'label': f"{final_g}_{final_c}"}

        # 영상 끝부분에서 등장한 객체도 카운팅 누락 없이 반영
        if tid not in counted_ids:
            counted_ids.add(tid)
            total_count += 1

    # 최종 결과 보고서 생성 및 리포트 윈도우 표시
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