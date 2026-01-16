import subprocess
import sys
import os

def install(package):
    print(f"📦 Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_torch_gpu():
    print("\n🚀 [Step 1] GPU 환경 감지 및 PyTorch 설치 시도...")
    try:
        # CUDA 12.1 버전용 PyTorch 설치 (최신 NVIDIA 환경 최적화)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        print("✅ GPU 버전 PyTorch 설치 완료!")
    except Exception as e:
        print(f"⚠️ GPU 버전 설치 실패: {e}")
        print("🔄 CPU 버전으로 설치를 시도합니다...")
        install("torch torchvision torchaudio")

def create_folders():
    print("\n📁 [Step 2] 프로젝트 필수 폴더 생성...")
    folders = ['logs', 'models', 'data', 'best_samples']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"   - 생성 완료: {folder}/")
        else:
            print(f"   - 이미 존재함: {folder}/")

def install_dependencies():
    print("\n📦 [Step 3] 추가 의존성 라이브러리 설치...")
    
    # 1. 일반 의존성 설치 (requirements.txt)
    if os.path.exists("requirements.txt"):
        print("   - requirements.txt 발견! 일괄 설치 진행...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
        except Exception as e:
            print(f"⚠️ requirements.txt 설치 중 오류 발생 (일부 패키지 실패 가능): {e}")
    else:
        print("   - requirements.txt 없음. 핵심 라이브러리 개별 설치...")
        pkgs = ["ultralytics", "opencv-python", "numpy", "supervision", "tqdm", "lapx"]
        for pkg in pkgs:
            install(pkg)

    # 2. OpenAI CLIP 설치 (Git 필수 - VLM 모드 핵심)
    print("\n   - OpenAI CLIP 라이브러리 설치 (VLM 엔진용)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/openai/CLIP.git"
        ])
        print("   ✅ CLIP 설치 완료!")
    except Exception as e:
        print(f"   ⚠️ CLIP 설치 실패: {e}")
        print("   👉 [Check] 시스템에 'Git'이 설치되어 있는지 확인해주세요.")
        print("   👉 Git 설치 후 다시 실행하거나, main_low.py(저사양 모드)를 이용하세요.")

def download_initial_models():
    print("\n📥 [Step 4] 기본 AI 모델 프리로딩 (YOLOv8)...")
    try:
        from ultralytics import YOLO
        # 메인 탐지 모델
        print("   - Downloading yolov8n.pt (Main Detector)...")
        YOLO("yolov8n.pt")
        # 저사양 성별 분류 모델
        print("   - Downloading yolov8n-cls.pt (Low-Spec Classifier)...")
        YOLO("yolov8n-cls.pt")
        
        # 모델 파일 이동 (.pt 파일이 루트에 생기면 models/ 폴더로 이동)
        for model_file in ["yolov8n.pt", "yolov8n-cls.pt"]:
            if os.path.exists(model_file):
                # 기존 파일이 있으면 덮어쓰기 위해 삭제 후 이동
                dest = os.path.join("models", model_file)
                if os.path.exists(dest): os.remove(dest)
                os.replace(model_file, dest)
                
        print("✅ 모델 준비 완료!")
    except Exception as e:
        print(f"⚠️ 모델 다운로드 중 오류 발생 (무시 가능/자동 다운로드 됨): {e}")

def main():
    print("================================================")
    print("   AI Tracking System - Environment Setup")
    print("================================================\n")
    
    # 1. 폴더 생성 (가장 먼저 수행)
    create_folders()
    
    # 2. PyTorch 설치
    install_torch_gpu()
    
    # 3. 기타 라이브러리 및 CLIP 설치
    install_dependencies()
    
    # 4. 모델 미리 받기
    download_initial_models()
            
    print("\n" + "="*50)
    print("🎉 모든 환경 설정이 완료되었습니다!")
    print("▶ 고사양 모드 실행: start.bat (또는 python main.py)")
    print("▶ 저사양 모드 실행: start-low.bat (또는 python main_low.py)")
    print("="*50)
    
    # 창이 바로 꺼지는 것을 방지 (평가자 확인용)
    os.system("pause")

if __name__ == "__main__":
    main()