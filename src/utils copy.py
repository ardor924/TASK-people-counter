import os
import cv2
import numpy as np
from datetime import datetime

def get_device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU (Intel/AMD)"
    except:
        return "Unknown Device"

def create_directories():
    for folder in ["logs", "best_samples", "models", "data"]:
        if not os.path.exists(folder): os.makedirs(folder)

def save_best_samples(video_name, best_crops_store):
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

# [수정됨] avg_c_conf (색상 평균 신뢰도) 인자 추가
def generate_report(video_name, total_count, gender_stats, avg_g_conf, avg_c_conf, fps, inference_log, file_prefix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_header = f"{file_prefix}report" if file_prefix else "report"
    report_filename = f"logs/{filename_header}_{video_name}_{timestamp}.txt"
    
    device_name = get_device_name()

    male_cnt = gender_stats.get('Male', 0)
    female_cnt = gender_stats.get('Female', 0)
    unk_cnt = gender_stats.get('Unk', 0) + gender_stats.get('Unknown', 0)
    analyzed_total = male_cnt + female_cnt + unk_cnt

    content = f"""
===================================================================================================================
[FINAL SYSTEM EVALUATION REPORT]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : {video_name}.mp4
    - Processed Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Storage Path   : {report_filename}
    - Hardware       : {device_name}
    - System Mode    : {'Low-Spec / Hybrid' if file_prefix else 'Standard / VLM'}

 2. TRAFFIC & ATTRIBUTE SUMMARY
    - ROI Line Crossed (진입 인원) : {total_count} 명
    - Total Analyzed   (분석 인원) : {analyzed_total} 명
    - Gender Distribution : Male({male_cnt}) / Female({female_cnt}) / Unk({unk_cnt})
    
    [Confidence Analysis]
    - Mean Conf (Gender)  : {avg_g_conf:.2f}% (Analysis Reliability)
    - Mean Conf (Color)   : {avg_c_conf:.2f}% (Semantic Reliability)

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

# [수정됨] avg_c_conf 인자 및 UI 표시 추가
def show_summary_window(video_name, total_count, gender_stats, avg_g_conf, avg_c_conf, fps):
    # 창 크기 세로 확장
    summary_bg = np.zeros((500, 600, 3), dtype=np.uint8)
    summary_bg[:] = (40, 40, 40)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    white, green, yellow, cyan = (255, 255, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0)
    
    male_cnt = gender_stats.get('Male', 0)
    female_cnt = gender_stats.get('Female', 0)
    unk_cnt = gender_stats.get('Unk', 0) + gender_stats.get('Unknown', 0)
    analyzed_total = male_cnt + female_cnt + unk_cnt

    cv2.putText(summary_bg, "FINAL SYSTEM SUMMARY", (50, 50), font, 1, yellow, 2)
    cv2.line(summary_bg, (50, 65), (550, 65), (150, 150, 150), 1)
    
    cv2.putText(summary_bg, f"Video: {video_name}", (50, 110), font, 0.65, white, 1)
    
    cv2.putText(summary_bg, f"ROI Crossed: {total_count}", (50, 150), font, 0.8, green, 2)
    cv2.putText(summary_bg, f"Total Analyzed: {analyzed_total}", (50, 190), font, 0.8, cyan, 2)
    
    cv2.putText(summary_bg, f"Gender: Male({male_cnt}) / Female({female_cnt})", (50, 230), font, 0.65, white, 1)
    
    # [추가] 신뢰도 표시 (Gender / Color)
    cv2.putText(summary_bg, f"Mean Conf (Gender): {avg_g_conf:.2f}%", (50, 270), font, 0.65, white, 1)
    cv2.putText(summary_bg, f"Mean Conf (Color) : {avg_c_conf:.2f}%", (50, 310), font, 0.65, white, 1)
    
    cv2.putText(summary_bg, f"Avg FPS: {fps:.1f}", (50, 350), font, 0.65, white, 1)
    
    cv2.putText(summary_bg, "Report & Best Samples saved.", (50, 420), font, 0.5, (200, 200, 200), 1)
    cv2.putText(summary_bg, "Press any key to exit...", (50, 450), font, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Final Summary Report", summary_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()