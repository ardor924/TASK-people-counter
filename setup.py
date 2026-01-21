import subprocess
import sys
import os

def install(package):
    print(f"📦 Installing {package}...")
    
    # [수정됨] 핵심 수정 파트!
    # 원본 문제: package가 "pkg1 pkg2" 처럼 띄어쓰기로 들어오면, 
    #            리스트의 마지막 요소가 ["...install", "pkg1 pkg2"] 가 되어 에러 발생.
    # 해결: .split()을 사용하여 공백 기준으로 문자열을 쪼개서 리스트로 합쳐줘야 함.
    #       결과 -> ["...install", "pkg1", "pkg2"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())
    except subprocess.CalledProcessError:
        print(f"⚠️ {package} 설치 중 문제가 발생했지만, 이미 설치되었거나 호환성 문제일 수 있어 넘어갑니다.")

def install_torch_gpu():
    print("\n🚀 [Step 1] GPU 환경 감지 및 PyTorch 설치 시도...")
    try:
        # CUDA 12.1 버전용 PyTorch 설치 (최신 NVIDIA 환경 최적화)
        # 여기는 원래부터 리스트로 잘 분리되어 있어서 문제가 없었습니다.
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        print("✅ GPU 버전 PyTorch 설치 완료!")
    except Exception as e:
        print(f"⚠️ GPU 버전 설치 실패: {e}")
        print("🔄 CPU 버전으로 설치를 시도합니다...")
        
        # [참고] 여기서 CPU 버전을 설치할 때 위에서 수정한 install 함수를 호출합니다.
        # 기존에는 여기서 "torch torchvision torchaudio" 문자열 때문에 에러가 났습니다.
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
        # 여기 있는 패키지들도 install 함수를 통해 안전하게 설치됩니다.
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
    print("================================================")
    
    # 0. 중복 설치 방지 체크 로직
    skip_install = False
    try:
        print("\n🔍 [Check] 필수 라이브러리 설치 상태 확인 중...")
        import torch
        import cv2
        import ultralytics
        
        print("✅ 핵심 라이브러리(Torch, OpenCV, YOLO)가 이미 설치되어 있습니다.")
        print("🚀 무거운 설치 과정을 건너뛰고 실행 준비를 마무리합니다.")
        skip_install = True
    except ImportError:
        print("⚠️ 필수 라이브러리가 감지되지 않았습니다. 전체 설치를 진행합니다.")
        skip_install = False

    # 1. 폴더 생성 (항상 실행)
    create_folders()
    
    if not skip_install:
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
    print("🎉 아무 키나 클릭하면 바로 영상이 실행됩니다.")
    print("="*50)
    
    os.system("pause")

if __name__ == "__main__":
    main()