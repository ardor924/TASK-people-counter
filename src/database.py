import sqlite3
import csv
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="people_counter.db"):
        self.db_path = os.path.join("data", db_name)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        self.create_table()
        self.create_summary_table()

    def create_table(self):
        # [수정] conf(신뢰도) 컬럼 추가 (실수형 REAL)
        query = """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            track_id INTEGER,
            gender TEXT,
            gender_conf REAL,
            clothing_color TEXT,
            color_conf REAL
        )
        """
        self.cursor.execute(query)
        self.conn.commit()

    def create_summary_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS video_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT,
            start_time TEXT,
            end_time TEXT,
            total_count INTEGER
        )
        """
        self.cursor.execute(query)
        self.conn.commit()

    def insert_log(self, track_id, gender, gender_conf, color, color_conf):
        # [수정] 신뢰도 값도 함께 저장
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
        INSERT INTO logs (timestamp, track_id, gender, gender_conf, clothing_color, color_conf) 
        VALUES (?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(query, (now, track_id, gender, gender_conf, color, color_conf))
        self.conn.commit()

    def save_summary(self, video_name, start_time, total_count):
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
        INSERT INTO video_summary (video_name, start_time, end_time, total_count)
        VALUES (?, ?, ?, ?)
        """
        self.cursor.execute(query, (video_name, start_time, end_time, total_count))
        self.conn.commit()

    def export_to_csv(self, video_name):
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        
        file_timestamp = datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")
        clean_name = os.path.splitext(os.path.basename(video_name))[0]

        # 1. Logs Export
        try:
            self.cursor.execute("SELECT * FROM logs")
            rows = self.cursor.fetchall()
            if rows:
                filename = f"Log_{clean_name}_{file_timestamp}.csv"
                csv_path = os.path.join(save_dir, filename)
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 헤더 수정
                    writer.writerow(['ID', 'Timestamp', 'Track_ID', 'Gender', 'Gender_Conf', 'Color', 'Color_Conf'])
                    writer.writerows(rows)
                print(f"💾 [Export] Logs saved to: {csv_path}")
        except Exception as e:
            print(f"⚠️ Export Logs Error: {e}")

        # 2. Summary Export
        try:
            self.cursor.execute("SELECT * FROM video_summary")
            rows = self.cursor.fetchall()
            if rows:
                filename = f"Summary_{clean_name}_{file_timestamp}.csv"
                csv_path = os.path.join(save_dir, filename)
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Video Name', 'Start Time', 'End Time', 'Total Count'])
                    writer.writerows(rows)
                print(f"💾 [Export] Summary saved to: {csv_path}")
        except Exception as e:
            print(f"⚠️ Export Summary Error: {e}")

    def print_recent_logs(self, limit=5):
        print("\n🔎 [DB Check] Recent 5 Detection Logs:")
        print("-" * 80)
        self.cursor.execute(f"SELECT * FROM logs ORDER BY id DESC LIMIT {limit}")
        rows = self.cursor.fetchall()
        if not rows:
            print("   (No data found)")
        for row in rows:
            # row 인덱스가 늘어났으므로 맞춰서 출력
            print(f"   pk:{row[0]} | ID:{row[2]} | {row[3]}({row[4]:.2f}) | {row[5]}({row[6]:.2f})")
        print("-" * 80)

    def close(self):
        self.conn.close()