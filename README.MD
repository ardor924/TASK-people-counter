# Attribute-based People Counter with ReID (Single Camera)

## Project Overview

본 프로젝트는 **단일 카메라 환경에서 특정 속성(옷 색상 등)을 기준으로 사람을 선별 카운팅하는 AI 기반 People Counter 프로토타입**입니다.  
객체 탐지(YOLOv8), 추적(ByteTrack/BoT-SORT), 속성 추론(Color Extraction), ROI 기반 카운팅까지의 **전체 파이프라인을 실무 적용 관점에서 설계**하는 것을 목표로 합니다.

본 과제는 단순한 데모 구현이 아닌,  
**실제 현장(Edge / Local 환경)에서 바로 확장·적용 가능한 구조와 최적화(FPS 방어)**를 중점적으로 고려하여 진행하였습니다.

---

## Demo

- 📹 **Demo Video** 👉 *(추후 업로드 예정)* - Real-time FPS Monitoring  
  - ROI(Region of Interest) based Counting  
  - Attribute Overlay (ID | Color)

---

## Table of Contents

1. Project Motivation & Design Intent  
2. System Architecture  
3. Dataset Description  
4. Attribute Definition  
5. Installation (**Updated**)  
6. Execution  
7. Project Structure  
8. Limitations & Future Work

---

## 1. Project Motivation & Design Intent

본 과제는 다음과 같은 실무 요구사항을 가정하고 설계되었습니다.

- **단일 카메라 환경 (Single Camera):** 중복 투자가 어려운 소규모 매장/거리 환경 고려
- **안정적인 추적 (Robust Tracking):** 다수의 인파 속에서도 ID 스위칭 최소화
- **엣지 최적화 (Edge Optimization):** 불필요한 연산을 줄여 4070Ti 기준 80+ FPS, 엣지 디바이스에서도 구동 가능한 경량 로직 구현
- **설치 편의성 (Easy Setup):** `setup.py`를 통한 원클릭 환경 구성

---

## 2. System Architecture

전체 파이프라인은 아래와 같은 흐름으로 구성됩니다.

1. **Person Detection:** YOLOv8 기반 사람 객체 탐지
2. **Tracking:** 프레임 간 ID 유지 (ReID)
3. **Attribute Inference (Caching):** - 매 프레임 연산하지 않고, ID별로 캐싱(Caching)하여 연산 부하 최소화
   - ROI 내 객체 또는 일정 주기로만 속성(옷 색상) 업데이트
4. **ROI Filtering:** 사용자가 지정한 관심 영역(ROI) 내 객체만 유효 타겟으로 선정
5. **Visualization:** 성능 저하를 막기 위한 Display Resizing 및 직관적인 오버레이

---

## 3. Dataset Description

### 3.1 Original Dataset
본 프로젝트에서는 **MOT Challenge 2016 (MOT16)** 데이터셋을 사용하였습니다.

| Dataset | Usage | Features |
|:---:|:---:|:---|
| **MOT16-09** | **Development** | 주간, 고정 카메라, 45도 앵글, 적절한 밀도 (속성 인식 개발 최적) |
| **MOT16-04** | **Evaluation** | 야간, 고정 카메라, 고밀도 인파 (추적 안정성 및 극한 환경 테스트) |

### 3.2 File Naming Convention
원본 데이터셋과의 명확한 대응 관계를 위해 아래와 같이 파일명을 변경하여 사용합니다.

- `MOT16-09` → `data/dev_day.mp4`
- `MOT16-04` → `data/eval_night.mp4`

---

## 4. Attribute Definition

본 프로젝트에서는 **실시간성**을 위해 무거운 Deep Learning Classifier 대신, 빠르고 효과적인 **Computer Vision 기반 접근법**을 사용했습니다.

- **Clothing Color (Upper Body):**
  - 객체의 상체 영역(Top 50%) 크롭
  - HSV 색공간 변환 및 픽셀 빈도 분석
  - 주요 색상(Red, Blue, Green, Black, White 등) 분류
  - **최적화:** 분석 시 이미지를 리사이징(Resize)하여 속도 10배 향상

---

## 5. Installation

본 프로젝트는 하드웨어 환경(CPU vs GPU)에 따라 PyTorch 설치 버전이 달라질 수 있어, 이를 자동으로 처리하는 설치 스크립트를 제공합니다.

### 5.1 Environment
- **Python:** 3.9+
- **OS:** Linux / Windows / macOS
- **GPU:** NVIDIA GPU (CUDA 12.x 권장) 또는 CPU

### 5.2 Method 1: 간편 설치 (Recommended)
`setup.py` 스크립트를 실행하면 **시스템의 GPU 사용 가능 여부를 자동으로 감지**하여 최적의 PyTorch 버전(CUDA 12.1 등)과 필수 라이브러리를 한 번에 설치합니다.

```bash
# 프로젝트 루트 경로에서 실행
python setup.py
```

> **Note:** > - NVIDIA GPU가 감지되면 `cu121`(CUDA 12.1) 기반의 PyTorch를 우선 설치합니다.
> - GPU가 없거나 설치에 실패할 경우, 자동으로 CPU 전용 버전으로 전환하여 설치를 완료합니다.

### 5.3 Method 2: 수동 설치 (Manual)
자동 설치에 실패하거나 특정 버전의 PyTorch가 필요한 경우 아래 절차를 따르세요.

**Step 1. PyTorch 설치**
본인의 환경에 맞는 버전을 먼저 설치합니다. (아래는 예시입니다)

- **NVIDIA GPU 사용자 (권장):**
  ```bash
  pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
  ```

- **CPU / Mac 사용자:**
  ```bash
  pip install torch torchvision torchaudio
  ```

**Step 2. 나머지 의존성 설치**
```bash
pip install -r requirements.txt
```

---

## 6. Execution

설치가 완료되면 아래 명령어로 실행할 수 있습니다.

### Basic Usage
```bash
python main.py --source data/dev_day.mp4
```

### Key Features during Execution
- **`q`**: 프로그램 종료
- **FPS Display**: 우측 상단에 실시간 처리 속도 표시 (노란색)
- **Attribute Info**: 각 객체 박스 위에 `ID | Color` 표시

---

## 7. Project Structure

```text
People-Counter-ReID/
│
├── data/                 # 테스트용 영상 파일
│   ├── dev_day.mp4
│   └── eval_night.mp4
├── models/               # YOLO 모델 가중치 저장소
│   └── yolov8n.pt
├── src/                  # 소스 코드 모듈
│   ├── __init__.py
│   ├── detector.py       # 객체 탐지 및 추적 (YOLO Wrapper)
│   ├── attributes.py     # 속성(색상) 분석 로직
│   └── tracker.py        # (Optional) 추가 추적 로직
│
├── main.py               # 메인 실행 파일
├── setup.py              # [New] 자동 설치 스크립트
├── requirements.txt      # 의존성 목록 (Torch 제외)
└── README.md             # 프로젝트 문서
```

---

## 8. Limitations & Future Work

- **야간 색상 인식:** 조도가 낮은 환경(MOT16-04)에서는 HSV 기반 색상 인식이 부정확할 수 있습니다. 향후 IR 카메라나 저조도 개선 알고리즘 도입이 필요합니다.
- **Deep Sort 통합:** 현재는 YOLO 내장 트래커를 사용 중이나, 더 정교한 ReID를 위해 DeepSort 또는 StrongSORT 통합을 고려할 수 있습니다.