import cv2
import numpy as np
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_and_save(self, video_name, start_time, end_time, loop_count, last_count, max_density, loop_results):
        """ 리포트 이미지 생성, 저장, 반환 """
        h, w = 500, 600
        report_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        gray = (200, 200, 200)
        cyan = (255, 255, 0)
        
        # Header
        cv2.putText(report_img, "Monitoring Session Report", (120, 50), font, 1, white, 2)
        cv2.line(report_img, (50, 70), (550, 70), white, 2)
        
        # Meta Info
        cv2.putText(report_img, f"Target: {video_name}", (50, 120), font, 0.6, white, 1)
        cv2.putText(report_img, f"Session Start: {start_time}", (50, 150), font, 0.6, white, 1)
        cv2.putText(report_img, f"Session End:   {end_time}", (50, 180), font, 0.6, white, 1)
        
        cv2.line(report_img, (50, 210), (550, 210), gray, 1)

        # Statistics
        cv2.putText(report_img, f"Total Loops: {loop_count}", (50, 250), font, 0.8, white, 1)
        
        cv2.putText(report_img, "Latest Result:", (50, 300), font, 0.8, yellow, 1)
        cv2.putText(report_img, f"- Count: {last_count} People", (70, 340), font, 1, green, 2)
        cv2.putText(report_img, f"- Max Density: {max_density}", (70, 380), font, 0.8, white, 1)
        
        # History
        cv2.line(report_img, (50, 410), (550, 410), gray, 1)
        cv2.putText(report_img, "[ History ]", (50, 440), font, 0.6, cyan, 1)
        
        history_text = ""
        if not loop_results:
            history_text = "Single Run Completed."
        else:
            items = [f"L{i+1}:{cnt}" for i, cnt in enumerate(loop_results)]
            history_text = " | ".join(items[-5:]) 
            if len(loop_results) > 5:
                history_text = "... " + history_text
                
        cv2.putText(report_img, history_text, (150, 440), font, 0.6, white, 1)

        # Save File
        clean_name = os.path.splitext(video_name)[0]
        file_timestamp = datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")
        filename = f"Report_{clean_name}_{file_timestamp}.jpg"
        save_path = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(save_path, report_img)
        print(f"📄 Report saved to: {save_path}")
        
        return report_img