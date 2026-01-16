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
    print("\n📦 [Step 3] 추가 의존성 라이브러리 설치 (requirements.txt)...")
    if os.path.exists("requirements.txt"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
    else:
        # requirements.txt가 없을 경우를 대비한 Fallback
        pkgs = ["ultralytics", "opencv-python", "numpy", "supervision", "tqdm", "lapx"]
        for pkg in pkgs:
            install(pkg)

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
                os.replace(model_file, os.path.join("models", model_file))
        print("✅ 모델 준비 완료!")
    except Exception as e:
        print(f"⚠️ 모델 다운로드 중 오류 발생 (무시 가능): {e}")

def main():
    print("================================================")
    print("   AI Tracking System - Environment Setup")
    print("================================================\n")
    
    # 1. 폴더 생성 (가장 먼저 수행)
    create_folders()
    
    # 2. PyTorch 설치
    install_torch_gpu()
    
    # 3. 기타 라이브러리 설치
    install_dependencies()
    
    # 4. 모델 미리 받기 (선택 사항이나 권장)
    download_initial_models()
            
    print("\n" + "="*50)
    print("🎉 모든 환경 설정이 완료되었습니다!")
    print("▶ 고사양 모드 실행: python main.py")
    print("▶ 저사양 모드 실행: python main_low.py")
    print("="*50)

if __name__ == "__main__":
    main()