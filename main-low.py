import cv2
import time
import numpy as np
import os
import sys
from collections import defaultdict, Counter
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import create_directories, save_best_samples, generate_report, show_summary_window

# =========================
# #1 Global Policy (Low-Spec)
# =========================
# VIDEO_PATH = "data/sample.avi"
VIDEO_PATH = "data/dev_day.mp4"
# VIDEO_PATH = "data/eval_day.mp4"
# VIDEO_PATH = "data/eval_night.mp4"
# VIDEO_PATH = "data/eval_indoors.mp4"
DETECTOR_PATH = "models/yolov8n.pt"
CLASSIFIER_PATH = "models/yolov8n-cls.pt"

RESIZE_WIDTH = 960       # 연산 속도 확보를 위한 리사이즈 해상도
CONF_THRESH = 0.6        # 객체 탐지 최소 확신도
DWELL_TIME_SEC = 0.5     # 카운팅 인정을 위한 최소 구역 체류 시간
FORGET_TIME_SEC = 0.7    # 트래킹 유실 시 ID 유지 시간
TRACK_BUFFER_SEC = 2.0   # 트래커 내부 데이터 유지 시간
ZONE_START_RATIO = 0.45  # 카운팅 기준선(ROI) 위치 비율

# ==========================================
# 2. Hybrid Engine (경량화 분석 엔진)
# ==========================================
class HybridAttributeEngine:
    """CPU 및 저사양 환경을 위해 딥러닝과 수학적 통계 기법을 혼합한 엔진"""
    def __init__(self, device="cuda"):
        from ultralytics import YOLO
        self.device = device
        
        # 성별 분류용 경량 모델 로드
        try:
            self.classifier = YOLO(CLASSIFIER_PATH).to(self.device)
            print(f"[INFO] Gender Classifier Loaded on {self.device}.")
        except Exception as e:
            print(f"[WARN] Failed to load classifier: {e}")
            self.classifier = None

        # 색상 판별을 위한 HSV 대표 중심값 정의 (Black, White, Gray 포함)
        self.color_centers = {
            "Black":  [0, 0, 30], "White":  [0, 0, 220], "Red":    [0, 200, 150],
            "Blue":   [115, 200, 150], "Yellow": [25, 200, 200], "Gray":   [0, 0, 100],
            "Green":  [60, 150, 100]
        }

    def analyze(self, crop_cv2):
        """객체 이미지로부터 성별(AI) 및 색상(K-means)을 동시 추출"""
        res_g, g_conf = "Unk", 0.0
        res_c, c_conf = "Unk", 0.0

        if crop_cv2.size == 0:
            return {"g": res_g, "c": res_c, "g_cf": 0, "c_cf": 0}

        # [성별 분석] 경량 Classification 모델 활용
        if self.classifier:
            try:
                res = self.classifier(crop_cv2, verbose=False, device=self.device)[0]
                top1_idx = res.probs.top1
                conf = res.probs.top1conf.item() * 100
                class_name = res.names[top1_idx].lower()
                # 클래스명 기반 성별 매핑
                if 'woman' in class_name or 'female' in class_name: res_g = 'Female'
                elif 'man' in class_name or 'male' in class_name: res_g = 'Male'
                else: res_g = "Male" if top1_idx % 2 != 0 else "Female"
                g_conf = conf
            except: pass

        # [색상 분석] K-means 군집화 알고리즘 활용 (수학적 통계)
        try:
            h, w = crop_cv2.shape[:2]
            # 상의(상의 20~50% 영역) 집중 추출
            roi = crop_cv2[int(h*0.2):int(h*0.5), int(w*0.3):int(w*0.7)]
            if roi.size > 0:
                pixels = roi.reshape((-1, 3)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 3 if len(pixels) >= 3 else 1
                # 이미지 내 주요 색상 3가지 추출
                _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                counts = np.bincount(labels.flatten())
                total = len(roi.reshape((-1, 3)))
                max_score = -1.0

                for i, center in enumerate(centers):
                    weight = counts[i] / total
                    if weight < 0.15: continue # 최소 분포량 미달 시 제외
                    
                    c_bgr = np.uint8([[center]])
                    c_hsv = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2HSV)[0][0]
                    # 살색(Skin tone) 노이즈 제거 필터
                    is_skin = (5 <= c_hsv[0] <= 25) and (40 <= c_hsv[1] <= 180)
                    
                    # 정의된 색상 중심점들과의 유클리드 거리 계산
                    min_dist = 9999
                    temp_c = "Unk"
                    for name, target in self.color_centers.items():
                        h_diff = abs(float(c_hsv[0]) - target[0])
                        if h_diff > 90: h_diff = 180 - h_diff
                        # 가중치 거리: Hue(2.5) > Saturation(1.0) > Value(0.8)
                        dist = np.sqrt(2.5*(h_diff**2) + 1.0*((float(c_hsv[1])-target[1])**2) + 0.8*((float(c_hsv[2])-target[2])**2))
                        if dist < min_dist:
                            min_dist = dist
                            temp_c = name
                    
                    # 최종 스코어링: 거리 점수 x 색상 분포량
                    score = (1000 - min_dist) * weight
                    if is_skin: score *= 0.3
                    if score > max_score:
                        max_score = score
                        res_c = temp_c
                        c_conf = max(0, min(100, (1 - min_dist/300) * 100))
        except: pass
        return {"g": res_g, "c": res_c, "g_cf": g_conf, "c_cf": c_conf}

# ==========================================
# 3. Main Pipeline (실행 프로세스)
# ==========================================
def main():
    import supervision as sv
    from ultralytics import YOLO

    create_directories()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running in Hybrid Voting Mode (Device: {device})")

    detector = YOLO(DETECTOR_PATH).to(device)
    hybrid_engine = HybridAttributeEngine(device=device)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}"); return

    cap = cv2.VideoCapture(VIDEO_PATH)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = min(raw_fps, 30.0) if raw_fps > 0 else 30.0
    frame_interval_ms = int(1000 / target_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # [객체 추적] ByteTrack 초기화
    tracker = sv.ByteTrack(lost_track_buffer=int(TRACK_BUFFER_SEC * target_fps))

    # 데이터 관리용 버퍼
    track_hits = defaultdict(int)
    id_map, next_display_id = {}, 1
    track_registry = {}
    zone_dwell, counted_ids, total_count = defaultdict(int), set(), 0
    
    # 속성 결정을 위한 시계열 투표 상자
    voting_box = defaultdict(lambda: {'g_scores': defaultdict(float), 'c_list': [], 'best_img': None, 'max_conf': 0, 'best_c_conf': 0.0})
    id_attributes, best_crops_store, inference_log = {}, {}, []
    conf_scores_g, conf_scores_c = [], []
    
    gender_stats = {"Male": 0, "Female": 0, "Unk": 0}
    frame_idx, start_time = 0, time.time()
    
    cv2.namedWindow("Hybrid Low-Spec AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid Low-Spec AI", RESIZE_WIDTH, int(RESIZE_WIDTH * 0.6))

    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 화면 해상도 조정 및 ROI 설정
        fh, fw = int(frame.shape[0] * (RESIZE_WIDTH/frame.shape[1])), RESIZE_WIDTH
        frame = cv2.resize(frame, (fw, fh))
        display = frame.copy()
        zone_y = int(fh * ZONE_START_RATIO)
        cv2.line(display, (0, zone_y), (fw, zone_y), (0, 0, 255), 2)

        # 객체 탐지 및 종횡비 필터링 (사람 형태 유효성 검증)
        results = detector(frame, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        valid_indices = [i for i, xyxy in enumerate(detections.xyxy) 
                         if (xyxy[3]-xyxy[1])/(xyxy[2]-xyxy[0]) > 1.15 and detections.confidence[i] >= CONF_THRESH]
        
        detections = detections[valid_indices]
        detections = tracker.update_with_detections(detections)

        active_now_ids = set()
        if detections.tracker_id is not None:
            for xyxy, raw_id, conf in zip(detections.xyxy, detections.tracker_id, detections.confidence):
                raw_id = int(raw_id)
                track_hits[raw_id] += 1
                if track_hits[raw_id] < 3: continue # 추적 초기 노이즈 배제

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # [ID Fusion Logic] 거리 기반 재식별 보정
                if raw_id not in id_map:
                    found_legacy = False
                    for disp_id, data in list(track_registry.items()):
                        if disp_id in active_now_ids: continue 
                        if frame_idx - data["last_frame"] > (FORGET_TIME_SEC * target_fps):
                            if disp_id in track_registry: del track_registry[disp_id]; continue
                        dist = np.sqrt((cx - data["pos"][0])**2 + (cy - data["pos"][1])**2)
                        if dist < (RESIZE_WIDTH * 0.02): # 이전 ID와 근접 시 동일인 판정
                            id_map[raw_id] = disp_id
                            found_legacy = True; break
                    if not found_legacy:
                        id_map[raw_id] = next_display_id; next_display_id += 1
                
                tid = id_map[raw_id]
                active_now_ids.add(tid)
                track_registry[tid] = {"pos": (cx, cy), "last_frame": frame_idx}
                
                # [분석 트리거] 3, 10, 20 프레임에서 다중 샘플링
                if tid not in id_attributes:
                    if track_hits[raw_id] in [3, 10, 20]:
                        crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
                        if crop.size > 0:
                            res = hybrid_engine.analyze(crop)
                            # 성별 확률 누적
                            if res['g'] in ['Male', 'Female']:
                                voting_box[tid]['g_scores'][res['g']] += res['g_cf']
                            # 색상 후보 누적
                            if res['c'] != "Unk":
                                voting_box[tid]['c_list'].append(res['c'])
                            # 최적 프레임(Best Frame) 갱신
                            curr_score = res['g_cf'] + res['c_cf']
                            if curr_score > voting_box[tid]['max_conf']:
                                voting_box[tid]['max_conf'] = curr_score
                                voting_box[tid]['best_c_conf'] = res['c_cf']
                                voting_box[tid]['best_img'] = crop.copy()
                                
                    # [최종 확정] 31프레임에서 투표 결과 집계
                    if track_hits[raw_id] == 31:
                        vdata = voting_box[tid]
                        m_score, f_score = vdata['g_scores']['Male'], vdata['g_scores']['Female']
                        
                        final_g = "Unk"
                        if (m_score + f_score) > 50: 
                            final_g = "Male" if m_score >= f_score else "Female"
                        
                        final_c = "Unk"
                        if vdata['c_list']: # 가장 빈번하게 등장한 색상 채택
                            final_c = Counter(vdata['c_list']).most_common(1)[0][0]
                        
                        id_attributes[tid] = {"g": final_g, "c": final_c}
                        gender_stats[final_g] = gender_stats.get(final_g, 0) + 1
                        
                        # 신뢰도 평균 기록
                        log_g_conf = max(m_score, f_score) / 3
                        log_c_conf = vdata['best_c_conf']
                        conf_scores_g.append(log_g_conf)
                        conf_scores_c.append(log_c_conf) 
                        
                        inference_log.append(f"[ID {tid:02d}] Final: {final_g}({log_g_conf:.1f}%) | {final_c}({log_c_conf:.1f}%)")
                        if vdata['best_img'] is not None:
                            best_crops_store[tid] = {'conf': log_g_conf, 'img': vdata['best_img'], 'label': f"{final_g}_{final_c}"}

                # [카운팅 로직] ROI 통과 및 체류 시간 검증
                attr = id_attributes.get(tid, {"g": "Scaning..", "c": "."})
                label = f"ID:{tid} | {attr['g'][0]}/{attr['c']}"

                if tid not in counted_ids and cy > zone_y:
                    zone_dwell[tid] += 1
                    if zone_dwell[tid] >= (DWELL_TIME_SEC * target_fps):
                        counted_ids.add(tid); total_count += 1

                # 바운딩 박스 및 라벨 출력
                color_box = (0, 255, 0) if tid in counted_ids else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(display, label, (x1, y1 - 10), 1, 0.7, color_box, 1)

        # HUD(Head-Up Display) 정보 출력
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, f"HYBRID VOTING SYSTEM: {os.path.basename(VIDEO_PATH)}", (20, 45), 1, 1.4, (0, 255, 255), 2)
        cv2.putText(display, f"MALE : {gender_stats['Male']} | FEMALE : {gender_stats['Female']}", (20, 95), 1, 1.2, (0, 255, 255), 1)
        
        real_fps = frame_idx / (time.time() - start_time)
        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(display, f"PROG : {int(progress)}% | FPS : {real_fps:.1f} (Limit: {target_fps:.0f})", (20, 135), 1, 0.9, (150, 150, 150), 1)

        cv2.imshow("Hybrid Low-Spec AI", display)
        
        # FPS 제어를 위한 대기 시간 계산
        proc_time_ms = (time.time() - loop_start_time) * 1000
        wait_ms = max(1, int(frame_interval_ms - proc_time_ms))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"): break

    cap.release(); cv2.destroyAllWindows()

    # 분석 종료 후 리포트 생성 및 데이터 저장
    elapsed = time.time() - start_time
    avg_g_conf = np.mean(conf_scores_g) if conf_scores_g else 0.0
    avg_c_conf = np.mean(conf_scores_c) if conf_scores_c else 0.0
    video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    
    save_best_samples(video_base_name, best_crops_store)
    generate_report(video_base_name, total_count, gender_stats, avg_g_conf, avg_c_conf, frame_idx/elapsed, inference_log, file_prefix="LOW_")
    show_summary_window(video_base_name, total_count, gender_stats, avg_g_conf, avg_c_conf, frame_idx/elapsed)

if __name__ == "__main__":
    main()