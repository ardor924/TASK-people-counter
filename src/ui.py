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

    def draw_main_ui(self, frame, detections, labels, sv_line_zone, fps):
        # 1. 궤적 & 박스
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        
        # 2. 라인 (시각효과용)
        if sv_line_zone:
            self.line_annotator.annotate(frame=frame, line_counter=sv_line_zone)
        
        # 3. 라벨 (검정 배경)
        for i, (xyxy, _, _, _, track_id, _) in enumerate(detections):
            if i < len(labels):
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                label_text = labels[i]
                (tw, th), _ = cv2.getTextSize(label_text, self.font, 0.5, 1)
                
                cv2.rectangle(frame, (x1, y1-25), (x1+tw+10, y1), (0, 0, 0), -1) 
                cv2.putText(frame, label_text, (x1+5, y1-8), self.font, 0.5, (0, 255, 0), 1)
        
        return frame
        
    def draw_overlay_text(self, frame, total_count, density, fps):
        """ [누락되었던 함수 추가됨] 정보 텍스트 오버레이 """
        h, w = frame.shape[:2]
        
        # 좌측 상단 박스
        cv2.rectangle(frame, (10, 10), (320, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Count: {total_count}", (25, 45), self.font, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Density: {density}", (25, 75), self.font, 0.8, (0, 255, 0), 2)

        # 우측 상단 FPS
        cv2.rectangle(frame, (w-180, 10), (w-10, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)}", (w-160, 45), self.font, 0.8, (0, 255, 255), 2)
        
        return frame

    def resize_for_display(self, frame):
        return cv2.resize(frame, (self.display_width, self.display_height))

    def save_results(self, log_data, video_name, start_time, total_count, max_density):
        os.makedirs("results", exist_ok=True)
        
        # CSV
        csv_path = f"results/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if log_data:
            pd.DataFrame(log_data).to_csv(csv_path, index=False)
            print(f"✅ CSV Saved: {csv_path}")
        
        # Report Image
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (800,600), (40,40,40), -1)
        cv2.rectangle(img, (0,0), (800,100), (0,150,0), -1)
        cv2.putText(img, "ANALYSIS REPORT", (240,65), self.font, 1.2, (255,255,255), 3)
        
        cv2.putText(img, f"Video: {video_name}", (50,180), self.font, 0.8, (220,220,220), 2)
        cv2.putText(img, f"Time : {start_time}", (50,230), self.font, 0.8, (220,220,220), 2)
        cv2.line(img, (50,280), (750,280), (150,150,150), 2)
        
        cv2.putText(img, "Total Count", (50,360), self.font, 1.0, (0,255,255), 2)
        cv2.putText(img, f"{total_count} Persons", (400,360), self.font, 1.3, (0,255,255), 3)
        
        cv2.putText(img, "Max Density", (50,440), self.font, 1.0, (100,255,100), 2)
        cv2.putText(img, f"{max_density} Persons", (400,440), self.font, 1.3, (100,255,100), 3)
        
        save_path = os.path.join("results", "final_report.png")
        cv2.imwrite(save_path, img)
        return img