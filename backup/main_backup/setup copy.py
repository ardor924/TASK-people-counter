# setup.py (프로젝트 루트에 생성)
import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_torch_gpu():
    print("🚀 GPU 환경 감지 중... (NVIDIA GPU 설치 시도)")
    try:
        # CUDA 12.1 버전 강제 설치 (Windows/Linux)
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

def main():
    print("📦 프로젝트 환경 설정을 시작합니다...")
    
    # 1. PyTorch 설치 (환경에 맞게 분기 처리 가능하나, 여기선 GPU 우선 시도)
    # 사용자가 직접 선택하게 할 수도 있음
    print("\n[Step 1] PyTorch 설치")
    install_torch_gpu()
    
    # 2. 나머지 라이브러리 설치
    print("\n[Step 2] 추가 의존성 설치 (requirements.txt)")
    if os.path.exists("requirements.txt"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
    else:
        # requirements.txt가 없을 경우 수동 설치
        pkgs = ["ultralytics", "opencv-python", "numpy", "tqdm", "lapx"]
        for pkg in pkgs:
            install(pkg)

def create_folders():
    folders = ['logs', 'models', 'data', 'best_samples']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 폴더 생성 완료: {folder}")

def main():
    print("📦 프로젝트 환경 설정을 시작합니다...")
    
    # 폴더 먼저 생성
    create_folders()
            
    print("\n🎉 모든 설치가 완료되었습니다! 'python main.py'를 실행하세요.")

if __name__ == "__main__":
    main()