# src/utils.py
import os
import cv2
import numpy as np
from datetime import datetime

def get_device_name():
    """현재 사용 중인 하드웨어 장치명 가져오기"""
    try:
        import torch
        if torch.cuda.is_available():
            # 예: NVIDIA GeForce RTX 4070 Ti
            return torch.cuda.get_device_name(0)
        return "CPU (Intel/AMD)"
    except:
        return "Unknown Device"

def create_directories():
    """필수 폴더 생성"""
    for folder in ["logs", "best_samples", "models", "data"]:
        if not os.path.exists(folder): os.makedirs(folder)

def save_best_samples(video_name, best_crops_store):
    """상위 5개 이미지 저장 (예외 처리 추가)"""
    if not best_crops_store:
        print("\n[INFO] No samples to save (No detections).")
        return

    print("\n[INFO] Saving Top 5 Best Samples...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        sorted_crops = sorted(best_crops_store.items(), key=lambda item: item[1]['conf'], reverse=True)[:5]
    except Exception as e:
        print(f"[WARN] Sorting failed: {e}")
        return

    for rank, (tid, data) in enumerate(sorted_crops, 1):
        try:
            img_filename = f"{video_name}_{timestamp}_Top{rank}_ID{tid:02d}_{data['label']}_{data['conf']:.0f}pct.jpg"
            save_path = os.path.join("best_samples", img_filename)
            cv2.imwrite(save_path, data['img'])
            print(f"   > Saved: {img_filename}")
        except Exception as e:
            print(f"   > Failed to save sample ID {tid}: {e}")

def generate_report(video_name, total_count, gender_stats, avg_conf, fps, inference_log):
    """텍스트 리포트 생성 (정확도 계산 로직 제거)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"logs/report_{video_name}_{timestamp}.txt"
    
    # [동적 장치명 감지]
    device_name = get_device_name()

    male_cnt = gender_stats.get('Male', 0)
    female_cnt = gender_stats.get('Female', 0)
    unk_cnt = gender_stats.get('Unknown', 0)

    content = f"""
===================================================================================================================
[FINAL SYSTEM EVALUATION REPORT]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : {video_name}.mp4
    - Processed Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Storage Path   : {report_filename}
    - Hardware       : {device_name}

 2. ATTRIBUTE DISTRIBUTION SUMMARY
    - Final Count (ROI)    : {total_count}
    - Gender Distribution  : Male({male_cnt}) / Female({female_cnt}) / Unk({unk_cnt})
    - Mean Conf (Gender)   : {avg_conf:.2f}% (Analysis Reliability)

 3. SYSTEM PERFORMANCE
    - Effective FPS      : {fps:.1f} (Running on {device_name})
    - Defense Strategy   : Aspect Ratio Filter(>1.5) + Best-Frame Sampling(3-step)
===================================================================================================================
"""
    print(content)
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(content)
            f.write("\n[APPENDIX: INDIVIDUAL OBJECT LOGS]\n")
            if inference_log:
                for log in sorted(inference_log): f.write(f"> {log}\n")
            else:
                f.write("> No objects detected.\n")
        print(f"[SUCCESS] Deep analysis report saved: {report_filename}")
    except Exception as e:
        print(f"[ERR] File I/O Error: {e}")

def show_summary_window(video_name, total_count, gender_stats, avg_conf, fps):
    """종료 시 요약 UI 표시"""
    summary_bg = np.zeros((400, 600, 3), dtype=np.uint8)
    summary_bg[:] = (40, 40, 40)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    white, green, yellow = (255, 255, 255), (0, 255, 0), (0, 255, 255)
    
    male_cnt = gender_stats.get('Male', 0)
    female_cnt = gender_stats.get('Female', 0)

    cv2.putText(summary_bg, "FINAL SYSTEM SUMMARY", (50, 50), font, 1, yellow, 2)
    cv2.line(summary_bg, (50, 65), (550, 65), (150, 150, 150), 1)
    
    cv2.putText(summary_bg, f"Video: {video_name}", (50, 110), font, 0.6, white, 1)
    cv2.putText(summary_bg, f"Total Count: {total_count}", (50, 150), font, 0.8, green, 2)
    cv2.putText(summary_bg, f"Gender: Male({male_cnt}) / Female({female_cnt})", (50, 190), font, 0.6, white, 1)
    cv2.putText(summary_bg, f"Mean Confidence: {avg_conf:.2f}%", (50, 230), font, 0.6, white, 1)
    cv2.putText(summary_bg, f"Avg FPS: {fps:.1f}", (50, 270), font, 0.6, white, 1)
    
    cv2.putText(summary_bg, "Report & Best Samples saved.", (50, 330), font, 0.5, (200, 200, 200), 1)
    cv2.putText(summary_bg, "Press any key to exit...", (50, 360), font, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Final Summary Report", summary_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()