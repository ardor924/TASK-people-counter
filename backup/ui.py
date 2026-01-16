import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import supervision as sv

class UIManager:
    def __init__(self, display_size=(960, 540)):
        self.display_width, self.display_height = display_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        # Supervision 버전에 따른 호환성 처리
        try:
            self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            self.use_new_sv = True
        except:
            self.use_new_sv = False
            
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    def draw_loading(self, frame_sample, dots=""):
        if frame_sample is None:
            img = np.zeros((self.display_height, self.display_width, 3), np.uint8)
        else:
            img = cv2.resize(frame_sample, (self.display_width, self.display_height))
            img = cv2.addWeighted(img, 0.3, np.zeros(img.shape, np.uint8), 0, 0)
        
        cv2.putText(img, f"System Initializing{dots}", (self.display_width//2 - 150, self.display_height//2), 
                   self.font, 1.0, (0, 255, 255), 2)
        return img

    def draw_main_ui(self, frame, detections, labels, sv_line_zone):
        # 1. 궤적 & 박스
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        
        # 2. 라인
        if sv_line_zone:
            self.line_annotator.annotate(frame=frame, line_counter=sv_line_zone)
        
        # 3. 라벨 (직접 그리기)
        for i, (xyxy, mask, conf, class_id, track_id, data) in enumerate(detections):
            if i < len(labels):
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                label_text = labels[i]
                (tw, th), _ = cv2.getTextSize(label_text, self.font, 0.5, 1)
                
                # 라벨 배경
                cv2.rectangle(frame, (x1, y1-25), (x1+tw+10, y1), (0, 0, 0), -1) 
                cv2.putText(frame, label_text, (x1+5, y1-8), self.font, 0.5, (0, 255, 0), 1)
        
        return frame
        
    def draw_overlay_text(self, frame, in_count, out_count, density, fps):
        """ [수정] Total 대신 IN/OUT 분리 표시 """
        h, w = frame.shape[:2]
        
        # 좌측 상단 박스 (IN/OUT, Density)
        cv2.rectangle(frame, (10, 10), (320, 90), (0, 0, 0), -1)
        
        # [변경] Total Count -> IN : X | OUT : Y
        info_text = f"IN: {in_count} | OUT: {out_count}"
        cv2.putText(frame, info_text, (25, 45), self.font, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Density: {density}", (25, 75), self.font, 0.8, (0, 255, 0), 2)

        # 우측 상단 FPS
        cv2.rectangle(frame, (w-180, 10), (w-10, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)}", (w-160, 45), self.font, 0.8, (0, 255, 255), 2)
        
        return frame

    def resize_for_display(self, frame):
        return cv2.resize(frame, (self.display_width, self.display_height))

    def save_results(self, log_data, video_name, start_time, in_count, out_count, max_density):
        os.makedirs("results", exist_ok=True)
        
        # CSV 저장
        csv_path = f"results/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if log_data:
            pd.DataFrame(log_data).to_csv(csv_path, index=False)
            print(f"✅ CSV Saved: {csv_path}")
        
        # 리포트 이미지 생성
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (800,600), (40,40,40), -1)
        cv2.rectangle(img, (0,0), (800,100), (0,150,0), -1)
        cv2.putText(img, "ANALYSIS REPORT", (240,65), self.font, 1.2, (255,255,255), 3)
        
        cv2.putText(img, f"Video: {video_name}", (50,180), self.font, 0.8, (220,220,220), 2)
        cv2.putText(img, f"Time : {start_time}", (50,230), self.font, 0.8, (220,220,220), 2)
        cv2.line(img, (50,280), (750,280), (150,150,150), 2)
        
        # [변경] Total -> IN / OUT 분리
        cv2.putText(img, "Total Count", (50,340), self.font, 1.0, (0,255,255), 2)
        cv2.putText(img, f"IN: {in_count}  |  OUT: {out_count}", (300,340), self.font, 1.1, (0,255,255), 3)
        
        cv2.putText(img, "Max Density", (50,420), self.font, 1.0, (100,255,100), 2)
        cv2.putText(img, f"{max_density} Persons", (300,420), self.font, 1.1, (100,255,100), 3)
        
        save_path = os.path.join("results", "final_report.png")
        cv2.imwrite(save_path, img)
        return img