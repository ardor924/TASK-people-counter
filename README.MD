#  프로젝트 개요
본 프로젝트는 **단일 카메라 환경** 에서 **VLM(CLIP)** 을 활용하여 **성별 및 의상 정보**를 정밀 분석하고 사람을 **카운팅** 하는 **AI 기반 SW** 입니다.

또한 제한된 시간 내에 모델의 신규 학습보다는, 최신 SOTA 모델인 **OpenAI CLIP**을 YoloV8n과 SuperVision에 통합하여 **성별(Gender)** 과  **상의 색상(Top Color)** 을 제로샷(Zero-shot)으로 추론하며,  **ID Fusion** 기술을 통해 추적 안정성을 극대화하여 실제 배포 가능할 수준의 신뢰도를 확보하는 데 집중해서 개발했습니다.

---
# 데모 시연 영상
-> 📹 **Demo Video** 👉 ![#링크] 

---

# 목차 
1. [Project Motivation](#-1-Project-Motivation)
2. [시스템 아키텍쳐](#2-시스템-아키텍쳐)
3. [데이터셋 정의](#3-데이터셋-정의(Dataset-Description))
4. [속성 정의](#4-속성-정의)
5. [Project Structure](#5-Project-Structure)
6. [설치 및 실행](#6-설치-및-실행)
7. [Accuracy & Reliability](#7-Accuracy-&-Reliability)
8. [차별화 포인트](#8-차별화-포인트)
9. [Technical & Future Work](#9-Technical-&-Future-Work)

# 1. Project Motivation

본 프로젝트는 다음과 같은 요구사항을 가정하고 설계되었습니다.

- **단일 카메라 환경 :**  소규모 공간/거리 환경 고려
- **Robust Tracking :** 다수의 인파 속에서도 ID 스위칭 최소화
- **하드웨어 최적화 :** 불필요한 연산을 줄여 4070Ti 기준 15+ FPS 퍼포먼스 
	- 추후 초저사양 디바이스에서도 구동 가능할것을 고려하여 확장성 기대
- **설치 편의성 (Easy Setup) :** `setup.py`를 통한 원클릭 설치환경 구성

---
# 2. 시스템 아키텍쳐

1. **Person Detection:** YOLOv8n 기반 고속 사람 탐지
    
2. **Robust Tracking:** ByteTrack 알고리즘을 통한 실시간 ID 유지
    
3. **ID Fusion Logic:** 시공간 제약 및 유클리드 거리 기반 ID 급증 현상 억제
    
4. **VLM Attribute Analysis:** CLIP 모델을 통한 성별 및 의상 색상(Black, White, Blue, Red, Gray, Yellow) 정밀 추론

5. **ROI Line Counting:** ROI 경계면 통과 시 체류 시간(Dwell Time) 검증 후 최종 카운트 반영
---
# 3.데이터셋 정의(Dataset Description)
### 3-1 원본 데이터셋
본 프로젝트에서는 [MOT Challenge 2016 (MOT16)](https://motchallenge.net/data/MOT16/) 데이터셋 을 사용하였습니다.

|   Dataset    |      Usage      | Features                                |
| :----------: | :-------------: | :-------------------------------------- |
| **MOT16-09** | **Development** | 주간, 고정 카메라, 적절한 밀도 (속성 인식 개발 최적)        |
| **MOT16-04** | **Evaluation**  | 야간, 고정 카메라, 고밀도 인파 (추적 안정성 및 극한 환경 테스트) |
| **MOT16-03** | **Evaluation**  | 실내, 동적 카메라, 적정 밀도 (추적 안정성 및 극한 환경 테스트)  |

### 3-2 파일명 변경규칙
원본 데이터셋과의 명확한 대응 관계를 위해 아래와 같이 파일명을 변경하여 사용합니다.
- `MOT16-09` → `data/dev_day.mp4`
- `MOT16-04` → `data/eval_night.mp4`
- `MOT16-03` → `data/eval_indoors.mp4`


---
# 4.  속성 정의 (Attribute Definition)

| **Attribute** | **Logic**              | **Keywords / Prompts**                                |
| ------------- | ---------------------- | ----------------------------------------------------- |
| **G(Gender)** | **Ensemble Sentences** | `a man with short hair`, `a woman with long hair` 등   |
| **C(Color)**  | **Semantic Matching**  | `black clothing`, `blue clothing`, `white clothing` 등 |
| **Decision**  | **Max-Confidence**     | 3회 추론 중 신뢰도 점수가 가장 높은 프레임의 결과 채택                      |

---
# 5. Project Structure

```text
TASK-people-counter/
│
├── data/                 # 개발 & 테스트용 영상 파일
│   ├── dev_day.mp4       # 주간 실외 / 고정 카메라
│   ├── eval_night.mp4    # 야간 실외 / 고정 카메라
│   └── eval_indoors.mp4  # 실내     / 동적 카메라 
│
├── models/               # YOLO,CLIP 모델 가중치 저장소
│   ├── yolov8n.pt
│   ├── yolov8n-cls.pt
│   └── ViT-B-16.pt
│
├── logs/                 # 영상종료 결과 로그 보관소
│   └── report_.txt       
│
├── best_samples/         # 신뢰도 상위 크롭 이미지 보관소
│   └── name_date_.jpg     
│
├── src/                  # 모듈 패키지
│   ├── __init__.py       # (빈 파일) 패키지 인식용
│   ├── config.py         # 설정 값 모음
│   ├── engine.py         # VLM 분석 엔진 (VLMAttributeEngine)
│   └── utils.py          # 레포트결과 저장 및 UI 표시 로직
│
├── main.py               # 메인 실행 파일
├── setup.py              # [New] 자동 설치 스크립트
├── requirements.txt      # 의존성 목록 (Torch 제외)
├── README.md             # 프로젝트 문서
└── start.bat             # 간편 영상실행
```


---

# 6. 설치 및 실행

본 프로젝트는 평가자의 편의를 위해 **원클릭 라이브러리 및 모델 구성** 기능을 탑재하고 있습니다.

#### 💻 개발 환경 (Tested Environment)

- **Python:** 3.11.14
- **Framework:** PyTorch 2.0 (CUDA 12.1)
- **Hardware:** RTX 4070Ti (12GB) / RAM 48GB (DDR4)
- **OS:** Windows 10 (Local PC)

#### 📋 시스템 요구 사양
|**구분**|**권장 사양 (VLM 모드)**|**최소 사양 (저사양 모드)**|
|---|---|---|
|**Python**|3.11 ~ 3.12 (Conda 환경 권장)|3.8+|
|**GPU**|NVIDIA RTX 30 시리즈 이상|Intel/AMD 내장 그래픽 가능|
|**RAM**|16GB 이상 (32GB 최적)|8GB 이상|
|**Storage**|5GB 이상의 여유 공간|5GB 이상의 여유 공간|

### 6-1. 프로젝트 다운로드 (Clone or Download)
> **Pre-requisite:** 시스템에 **Git**이 설치되어 있어야 원격 클론이 가능합니다.

현재 깃허브 레포지토리에서 프로젝트를 clone 또는 zip으로 다운받아주세요

```bash 
# 레포지토리 클론 
git clone https://github.com/ardor924/TASK-people-counter
# 폴더 이동
cd TASK-people-counter
```

### 6-2. 소스 영상 준비
하단 링크에서 테스트 영상을 다운로드하여 `data/` 폴더에 넣어주세요. (미설정 시 기본 샘플로 실행됩니다.)

- **영상 출처:** [MOT Challenge 2016](https://motchallenge.net/data/MOT16/)
- **라이선스:** CC BY-NC-SA 4.0
- **파일링크**
	▶ [dev_day.mp4(MOT16-09) :  주간-카메라고정 실외영상](#)
	▶ [eval_day.mp4(MOT16-04) : 야간-카메라고정 실외영상(붐비는,원거리)](#)
	▶ [eval_indoors.mp4(MOT16-03) : 실내-비고정/동적 카메라 영상](#)

### 6-3 원클릭 간편설치&자동실행

**⚠️ 설치전 참고사항  
- Python 3.9+ 또는 Anaconda가 설치되어 있어야 합니다.
- **Python 3.13 사용자 주의:** 3.13은 아직 PyTorch CUDA 지원이 불안정하여 자동으로 CPU 버전이 설치될 수 있습니다. GPU 가속을 위해 **3.11~3.12** 환경을 권장합니다.
- 3.8 Python 이하 사용자의경우 Pytorch를 비롯한 라이브러리들이 정상작동이 안되 충돌이 날 수 있습니다.
##### 1. 일반 버전 (GPU/VLM 사용)
- `start.bat` 실행 시 가상환경 생성, 라이브러리(`YOLO`, `CLIP` 등) 및 모델 가중치 설치 후 즉시 `main.py`가 실행됩니다.
##### 2. 저사양 버전 (CPU/내장 그래픽용)
- `start-low.bat` 실행 시 경량화된 설정으로 `main-low.py`가 즉시 실행됩니다.
##### 3. Mac 사용자 
```Bash
chmod +x start-low-for-mac.sh
./start-low-for-mac.sh
```

---
### 6-4. IDE(VS Code/PyCharm) 환경에서의 수동 설치

#### **VS Code에서 진행 시**
1. **가상환경 인터프리터 설정:**
    - 프로젝트 폴더를 연 후 `Ctrl + Shift + P`를 누릅니다.
    - `Python: Select Interpreter`를 검색하여 선택합니다.
    - 미리 생성한 **Conda(ai_counter_env)** 또는 **venv(.venv)** 경로를 지정합니다.

2. **setup.py 실행:**
    - 왼쪽 파일 탐색기에서 `setup.py`를 우클릭한 후 **'Run Python File in Terminal'**을 선택합니다.
    - 모든 라이브러리와 모델 가중치가 선택된 가상환경에 자동으로 설치됩니다.

3. **메인 프로그램 실행:**
    - `main.py` 파일을 열고 동일하게 실행(F5 또는 우클릭 실행)합니다.
        
#### **PyCharm에서 진행 시**
1. **Interpreter 설정:** `Settings` > `Project` > `Python Interpreter`에서 가상환경을 추가/선택합니다.

2. **스크립트 실행:** `setup.py` 실행 후 설치가 완료되면 `main.py`를 실행합니다


---
### 6-5 수동설치 및 실행
> Conda 사용하지 않거나 환경을 직접 구축하는 경우 아래 단계를 따라주세요.

**1. 가상환경 생성 및 활성화**
```Bash
python -m venv .venv
# Windows
call .venv\Scripts\activate
# Mac
source .venv/bin/activate
```

**2. 환경 구축 스크립트 실행** 
PyTorch(CUDA 12.1), Ultralytics, CLIP 등을 일괄 설치하고 폴더 구조를 자동 생성합니다.
```Bash
python setup.py
```

**3. 프로그램 실행**
```Bash
python main.py  # 또는 python main-low.py
```

---
### 🚀 팁 (Tip)

- **가상환경 이름 주의:** 자동화 스크립트가 `ai_counter_env`라는 이름을 사용하므로, 수동 생성 시 이 이름과 겹치지 않게 주의해 주세요.
    
- **경로 변경:** 영상 파일 경로나 모델 설정을 바꾸려면 `main.py` 상단의 환경 설정 변수를 수정하세요.   

+ 영상이 종료될때 키보드에서 `q` 를 누르면 레포트도 종료할수있습니다.

+ Best_samples/ 폴더 에서 실행한 영상의 가장 신뢰도 높은 5명의 이미지가 저장되었으니 성능 확인용으로 참고할수 있습니다.

+ logs 폴더에 결과후 보고서를 저장해놓았습니다. 성능 확인용으로 참고할수 있습니다.

---
# 7. Accuracy & Reliability
Yolov8n + SuperVision + VLM(CLIP:ViT-B/16) 조합
- eval_day(주간실외 / 고정카메라) :

- eval_night(야간실외 / 고정카메라) :

- eval_indoors(실내/ 동적카메라) :

---

# 8. 차별화 포인트

|**차별화 기술**|**일반적인 방식**|**본 프로젝트 방식**|**기대 효과**|
|---|---|---|---|
|**속성 분석**|특정 데이터 학습 필요|**CLIP 제로샷 추론**|추가 학습 없이 다양한 복장 식별 가능|
|**분석 안정성**|단일 프레임 판단 (불안정)|**시계열 다수결 투표**|조명/가림 현상에 강한 정밀 데이터 확보|
|**추적 기술**|단순 거리 기반 매칭|**ID Fusion + ByteTrack**|밀집 환경에서도 ID 스위칭 억제|


---
# 9.  Technical & Future Work

### 9-1. 트러블슈팅 
야간 오탐지 문제를 해결하기 위해 기하학적 종횡비(Aspect Ratio) 필터링 알고리즘을 설계 및 적용했습니다.

### 9-2. 기술변경

❌ 초기 설정(특성활용에 휴리스틱방식)  : 
- Yolov8n + HSV + K-means
임베디드 혹은 저사양 하드웨어 환경을 고려하여 수학적 통계방식을 채택하였으나
특징(성별,의상색깔)에 인식률과 퍼포먼스가 저조하고 다양한 환경(광량, 노이즈, 구도)에서 범용성에 맞게 사용하기 어렵다고 판단하여 기술을 변경.

✅ 최종채택 (특성활용에 VLM채택) : 
- Yolov8n + SuperVision + VLM(CLIP:ViT-B/16)
과제 특성상  `SW의 우수성을 입증할 수 있어야 함 (정확성, 추론속도, 기능)` 과 5일한정 마감기간에 따른 판단으로, 모델에 특성을 따로 학습을 할수 없었기 때문에 성능과 효율을 타협하여 인식률을 우선시 하여 채택했습니다.


⚠ 단점 보완
최신 VLM SOTA 모델을 채택하여 임베디드 혹은 저사양 하드웨어 환경에서 작동이 어려울수 있으나 RTX 4070ti 12GB 기준 FPS15로 퍼포먼스 최적화도 신경써서 개발

### 9-3. 추후 도전과제 (Future Works)

현재는 저사양 하드웨어 환경에서 작동이 어려울수 있으나
임베디드나 저사양에서 작동할것을 고려하여 추후에 VLM(CLIP:ViT-B/16) 에
ONNX 혹은 NCNN과 양자화(INT8) 적용하여 최적화를 기대할수 있을것으로 예상

sqlite 등의 DB를 생성하고 연결하여 영상종료후 추출되는 인원의 정보를 활용할수 있을것으로 예상

개발 프로토타입 이므로 상용화및 실무 배포 하기 위해서 깔끔한 폰트와 UI/UX를 추가할 필요가 있음

---

