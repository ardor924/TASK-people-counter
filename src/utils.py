import os
import cv2
import numpy as np
from datetime import datetime

# ==========================================
# 1. 하드웨어 가속 환경 감지
# ==========================================
def get_device_name():
    """현재 시스템에서 사용 중인 하드웨어(GPU/CPU) 이름을 반환합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            # 실제 연산에 사용된 GPU 모델명을 정확히 리포트하기 위함
            return torch.cuda.get_device_name(0)
        return "CPU (Intel/AMD)"
    except:
        return "Unknown Device"

# ==========================================
# 2. 프로젝트 디렉토리 자동 구성
# ==========================================
def create_directories():
    """프로젝트 실행에 필요한 필수 폴더(로그, 샘플, 모델, 데이터)를 자동 생성합니다."""
    for folder in ["logs", "best_samples", "models", "data"]:
        if not os.path.exists(folder): 
            os.makedirs(folder)

# ==========================================
# 3. 최적 성능 샘플 이미지 저장 (우수성 입증용)
# ==========================================
def save_best_samples(video_name, best_crops_store):
    """
    분석된 객체 중 신뢰도가 가장 높은 상위 5개의 이미지를 추출하여 저장합니다.
    평가자가 AI의 판별 정확도를 육안으로 검토할 수 있는 근거 자료가 됩니다.
    """
    if not best_crops_store:
        print("\n[INFO] No samples to save (No detections).")
        return

    print("\n[INFO] Saving Top 5 Best Samples for Validation...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 신뢰도(Confidence) 기준 내림차순 정렬 후 상위 5개 슬라이싱
        sorted_crops = sorted(best_crops_store.items(), key=lambda item: item[1]['conf'], reverse=True)[:5]
    except Exception as e:
        print(f"[WARN] Sorting failed: {e}")
        return

    for rank, (tid, data) in enumerate(sorted_crops, 1):
        try:
            # 파일명에 비디오명, 순위, ID, 판별라벨(성별_색상), 신뢰도를 모두 포함하여 가시성 확보
            img_filename = f"{video_name}_{timestamp}_Top{rank}_ID{tid:02d}_{data['label']}_{data['conf']:.0f}pct.jpg"
            save_path = os.path.join("best_samples", img_filename)
            cv2.imwrite(save_path, data['img'])
            print(f"   > Saved: {img_filename}")
        except Exception as e:
            print(f"   > Failed to save sample ID {tid}: {e}")

# ==========================================
# 4. 최종 시스템 평가 보고서 생성 (Data Pipeline)
# ==========================================
def generate_report(video_name, total_count, gender_stats, avg_g_conf, avg_c_conf, fps, inference_log, file_prefix=""):
    """
    분석이 종료된 후 인구 통계, 모델 신뢰도, 시스템 성능을 포함한 
    최종 텍스트 리포트를 생성하여 'logs/' 폴더에 보관합니다.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_header = f"{file_prefix}report" if file_prefix else "report"
    report_filename = f"logs/{filename_header}_{video_name}_{timestamp}.txt"
    
    device_name = get_device_name()

    male_cnt = gender_stats.get('Male', 0)
    female_cnt = gender_stats.get('Female', 0)
    unk_cnt = gender_stats.get('Unk', 0) + gender_stats.get('Unknown', 0)
    analyzed_total = male_cnt + female_cnt + unk_cnt

    # 리포트 레이아웃 구성 (엔지니어링 전문성 강조)
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
    - Defense Strategy   : Aspect Ratio Filter(>1.15) + Best-Frame Sampling(3-step)
===================================================================================================================
"""
    print(content)
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(content)
            # 개별 객체에 대한 세부 추론 로그를 부록으로 추가
            f.write("\n[APPENDIX: INDIVIDUAL OBJECT LOGS]\n")
            if inference_log:
                for log in sorted(inference_log): 
                    f.write(f"> {log}\n")
            else:
                f.write("> No objects detected.\n")
        print(f"[SUCCESS] Deep analysis report saved: {report_filename}")
    except Exception as e:
        print(f"[ERR] File I/O Error: {e}")

# ==========================================
# 5. 시각적 요약 윈도우 표시 (UI/UX)
# ==========================================
def show_summary_window(video_name, total_count, gender_stats, avg_g_conf, avg_c_conf, fps):
    """사용자가 분석 결과를 즉시 확인할 수 있도록 별도의 UI 창을 생성하여 요약 정보를 표시합니다."""
    
    # 짙은 회색 배경 생성 (가독성 고려)
    summary_bg = np.zeros((500, 600, 3), dtype=np.uint8)
    summary_bg[:] = (40, 40, 40)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    white, green, yellow, cyan = (255, 255, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0)
    
    male_cnt = gender_stats.get('Male', 0)
    female_cnt = gender_stats.get('Female', 0)
    unk_cnt = gender_stats.get('Unk', 0) + gender_stats.get('Unknown', 0)
    analyzed_total = male_cnt + female_cnt + unk_cnt

    # UI 텍스트 배치
    cv2.putText(summary_bg, "FINAL SYSTEM SUMMARY", (50, 50), font, 1, yellow, 2)
    cv2.line(summary_bg, (50, 65), (550, 65), (150, 150, 150), 1)
    
    cv2.putText(summary_bg, f"Video: {video_name}", (50, 110), font, 0.65, white, 1)
    
    # 주요 카운팅 결과 강조
    cv2.putText(summary_bg, f"ROI Crossed: {total_count}", (50, 150), font, 0.8, green, 2)
    cv2.putText(summary_bg, f"Total Analyzed: {analyzed_total}", (50, 190), font, 0.8, cyan, 2)
    
    cv2.putText(summary_bg, f"Gender: Male({male_cnt}) / Female({female_cnt})", (50, 230), font, 0.65, white, 1)
    
    # 평균 신뢰도(VLM 성능 지표) 표시
    cv2.putText(summary_bg, f"Mean Conf (Gender): {avg_g_conf:.2f}%", (50, 270), font, 0.65, white, 1)
    cv2.putText(summary_bg, f"Mean Conf (Color) : {avg_c_conf:.2f}%", (50, 310), font, 0.65, white, 1)
    
    # 시스템 FPS 지표 표시
    cv2.putText(summary_bg, f"Avg FPS: {fps:.1f}", (50, 350), font, 0.65, white, 1)
    
    cv2.putText(summary_bg, "Report & Best Samples saved.", (50, 420), font, 0.5, (200, 200, 200), 1)
    cv2.putText(summary_bg, "Press any key to exit...", (50, 450), font, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Final Summary Report", summary_bg)
    cv2.waitKey(0) # 사용자 키 입력 시 창 닫기
    cv2.destroyAllWindows()