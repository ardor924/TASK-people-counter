import os
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))  # .../utils
project_root = os.path.dirname(current_dir)               # .../project_root
sys.path.append(project_root)

# 경로 설정 후 import
from backup.database import DatabaseManager

def reset_project():
    print(f"📍 Project Root Detected: {project_root}")
    print("🧹 [System Reset] 초기화를 시작합니다...")

    # 1. DB 파일 삭제
    db_path = os.path.join(project_root, "data", "people_counter.db")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"✅ DB 파일 삭제 완료: {db_path}")
        except Exception as e:
            print(f"⚠️ DB 삭제 실패: {e}")
    else:
        print("ℹ️ 삭제할 DB 파일이 없습니다.")

    # 2. Results 폴더 비우기
    results_dir = os.path.join(project_root, "results")
    if os.path.exists(results_dir):
        try:
            shutil.rmtree(results_dir) # 폴더 통째로 삭제
            print(f"✅ 결과 폴더 삭제 완료: {results_dir}")
        except Exception as e:
            print(f"⚠️ 결과 폴더 삭제 실패: {e}")
    
    # 3. 다시 생성 (DB 스키마 + 폴더)
    print("🔄 시스템 재구축 중...")
    
    # results 폴더 다시 생성
    os.makedirs(results_dir, exist_ok=True)
    
    # DB 매니저를 호출하여 테이블 재생성
    # (DatabaseManager 내부에서 data 폴더 경로는 상대경로로 되어있을 수 있으므로 주의 필요하지만,
    #  보통 main.py 실행 위치 기준이므로 여기서는 DB파일 생성만 확인하면 됨)
    try:
        # 작업 디렉토리를 잠시 루트로 변경 (DB 생성 위치 보정을 위해)
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        db = DatabaseManager()
        db.close()
        
        os.chdir(original_cwd) # 원복
        print("✅ DB 및 테이블 재생성 완료")
        
    except Exception as e:
        print(f"⚠️ DB 재생성 중 오류 발생: {e}")
    
    print("\n🎉 [Complete] 모든 데이터가 초기화되었습니다!")

if __name__ == "__main__":
    check = input("정말로 모든 데이터를 삭제하고 초기화하시겠습니까? (y/n): ")
    if check.lower() == 'y':
        reset_project()
    else:
        print("❌ 초기화가 취소되었습니다.")