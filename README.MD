#  프로젝트 개요

### " 원클릭으로 작동하는  사용자 맞춤형 AI 기반 SW"

본 프로젝트는 **단일 카메라 환경** 에서 **VLM(CLIP)** 을 활용하여 **성별 및 의상 정보**를 정밀 분석하고 사람을 **카운팅** 하는 **AI 기반 SW** 입니다.

가장 큰 특징은 **사용자의 하드웨어 환경(고사양 GPU vs 저사양 노트북)** 에 맞춰 최적의 분석 엔진을 선택할 수 있는 **'이원화된 아키텍처'**  와 어떤 환경에서도 **원클릭** 으로 실행되는 SW를 목표로 했습니다.

 또한 제한된 시간 내에 모델의 신규 학습보다는, 최신 SOTA 모델인 **OpenAI CLIP**을 YoloV8n과 SuperVision에 통합하여 **성별(Gender)** 과  **상의 색상(Top Color)** 을 **제로샷(Zero-shot)** 으로 추론하며,  **ID Fusion** 기술을 통해 추적 안정성을 향상시켜 실제 배포 가능할 수준의 신뢰도를 확보하는 데 집중해서 개발했습니다.

---
# 기능소개

1 원클릭 설치 및 실행 -> 사양별 OS별 맞춤 간편설치
2 타겟 영상내 인원 카운팅 -> Yolov8n
3 인원추적기술 -> supervision Bytracker
4 특성추출 -> VLM(CLIP) 자연어 조합 로직 + 제로샷
5 FPS방어 -> 
6 ROI 카운팅 개선
7 영상종료시 결과 레포트 제공
8 신뢰도 상위 5명을 육안으로 직접 성능 비교가능


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

### 2-1. 시스템 엔진별 기술 및 모델 비교표

| **구분**     | **일반 모드 (VLM Mode)**                                                        | **경량화 모드 (Hybrid Mode)**                                        |
| ---------- | --------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **타겟환경**   | 외장 GPU (RTX 30/40 시리즈)                                                      | CPU 및 내장 그래픽 (Laptop/Edge)                                      |
| **핵심 가치**  | **높은 분석 정밀도 및 범용성**                                                         | **빠른 추론 속도 및 효율성**                                              |
| **객체 탐지**  | **YOLOv8n** (Detection)                                                     | **YOLOv8n** (Detection)                                         |
| **객체 추적**  | **ByteTrack** (Supervision)                                                 | **ByteTrack** (Supervision)                                     |
| **추적 안정화** | **ID Fusion** (유클리드 거리 기반)                                                  | **ID Fusion** (유클리드 거리 기반)                                      |
| **성별 분석**  | **OpenAI CLIP (ViT-B/16)**<br><br>  <br><br>_(Semantic Prompt Matching)_    | **YOLOv8n-cls**<br><br>  <br><br>_(Lightweight Classification)_ |
| **의상 색상**  | **OpenAI CLIP (ViT-B/16)**<br><br>  <br><br>_(Zero-shot Semantic matching)_ | **HSV K-Means Clustering**<br><br>  <br><br>_(수학적 통계 기반 군집화)_   |
| **판단 로직**  | **Confidence Best-Selection**<br><br>  <br><br>_(최고 신뢰도 프레임 채택)_            | **Temporal Weighted Voting**<br><br>  <br><br>_(시계열 가중치 투표)_    |
| **특징**     | 추가 학습 없는 제로샷 추론                                                             | 저사양 환경 실시간성 확보                                                  |

---
# 3.데이터셋 정의(Dataset Description)
### 3-1 원본 데이터셋
본 프로젝트에서는 [MOT Challenge 2016 (MOT16)](https://motchallenge.net/data/MOT16/) 데이터셋의 일부를 사용하였습니다.

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

| **Attribute** | **Logic**                           | **Keywords / Prompts**                                     |
| ------------- | ----------------------------------- | ---------------------------------------------------------- |
| **G(Gender)** | **Ensemble Sentences**              | `a man with short hair`, `a woman with long hair` 등        |
| **C(Color)**  | **Semantic Matching**               | `black clothing`, `blue clothing`, `white clothing` 등      |
| **Decision**  | **Confidence-based Best Selection** | 샘플링된 3개 프레임 중 VLM 확신도(Softmax Score)가 가장 높은 결과를 최종 속성으로 채택 |

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
│   ├── yolov8n.pt        # YOLO 모델 다운로드(파일이 미리없어도 자동다운)
│   ├── yolov8n-cls.pt    
│   └── ViT-B-16.pt       # VLM 모델 다운로드(파일이 미리없어도 자동다운)
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

본 프로젝트는 평가자의 편의를 위해 **원클릭 라이브러리 및 모델 구성** 기능을 탑재하고 있습니다. 또한 환경 구축의 번거로움을 최소화하기 위해 여러 OS 대응 코드를 작성 했습니다.

#### 💻 개발 환경 (Tested Environment)

- **Python:** 3.11.14
- **Framework:** PyTorch 2.0 (CUDA 12.1)
- **Hardware:** RTX 4070Ti (12GB) / RAM 48GB (DDR4)
- **OS:** Windows 10 (Local PC)

테스트 디바이스 : 
RTX4060 PC | RTX2060 PC | RTX 3080 PC | inter graphic notebook 
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
하단 링크에서 테스트 영상을 다운로드하여 `data/` 폴더에 넣어주세요. (**영상이 없을시 SW가 실행되지 않으며**, 파일경로 미설정 시 기본 샘플로 실행됩니다.)

- **영상 출처:** [MOT Challenge 2016](https://motchallenge.net/data/MOT16/), [MOT Challenge 2016](https://motchallenge.net/data/MOT16/)
- **라이선스:** CC BY-NC-SA 4.0
- **파일링크**

	▶ sample.avi(TownCentreXVID) :  주간-카메라고정 실외영상 : 기본내장
	
	▶ [dev_day.mp4(MOT16-09) :  주간-카메라고정 실외영상](https://drive.google.com/uc?export=download&id=1zLggWjesXdZrG913pr9JdhvCqYgtWygr)	
	
	▶ [eval_day.mp4(Oxford Town Centre) :  주간-카메라고정 실외영상](https://drive.google.com/uc?export=download&id=10FdQ7iV6_I9FuJIQAT3X3ImvHl9f7LWF)
	
	▶ [eval_nitght.mp4(MOT16-04) : 야간-카메라고정 실외영상(붐비는,원거리)](https://drive.google.com/uc?export=download&id=1PAaZPVPciWOAVZ8t-A1qJy4_dvDwAG0i)
	
	▶ [eval_indoors.mp4(MOT16-03) : 실내-비고정/동적 카메라 영상](https://drive.google.com/uc?export=download&id=1ZKL2iqgjA5LUfd9dUvikreBxUsdDELLc)

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

- **성능 목표** : 
	- 실시간 분석 환경(15 FPS+)을 유지하면서 주간/야간 평균 85% 내외의 속성 인식률을 확보

- **신뢰도 기반 필터링**: 
	- 단일 프레임의 오판을 방지하기 위해 **3-step Best-Frame Sampling**을 수행하며, 그 중 **VLM 확신도(Confidence Score)가 가장 높은 최적 프레임**을 채택하여 속성 판별의 정확도를 보정
    
- **시스템 안정성**: 
	- 이번 프로젝트에서는 정확도 향상도 중요하지만 실제 현장에서 발생할 수 있는 ID 스위칭 및 다수 인파 상황에서의 시스템 안정성(Robustness) 확보에 초점을 맞춤


>※ Accuracy는 MOT16-09 Ground Truth(25명)를 기준으로 산출되었으며, 타 영상 실행 시 정답 수치에 따라 변동될 수 있습니다.

Yolov8n + SuperVision + VLM(CLIP:ViT-B/16) 조합
- dev_day(주간실외 / 고정카메라) :
```bash

===================================================================================================================
[FINAL SYSTEM EVALUATION REPORT]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : dev_day.mp4
    - Processed Date : 2026-01-16 09:33:16
    - Storage Path   : logs/report_dev_day_20260116_093316.txt

 2. ATTRIBUTE DISTRIBUTION SUMMARY
    - Final Count (ROI)    : 21
    - Gender Distribution  : Male(7) / Female(12) / Unk(0)
    - Mean Conf (Gender)   : 87.94% (Analysis Reliability)

 3. SYSTEM PERFORMANCE
    - Effective FPS      : 15.6 (Target 15+ on 4070Ti)
    - Defense Strategy   : Aspect Ratio Filter(>1.5) + Best-Frame Sampling(3-step)
===================================================================================================================

[APPENDIX: INDIVIDUAL OBJECT LOGS]
> [ID 01] Best G: 89.8% | C: 40.8% | Res: Male/Gray
> [ID 02] Best G: 65.8% | C: 30.7% | Res: Female/Black
> [ID 03] Best G: 96.9% | C: 91.8% | Res: Male/Blue
> [ID 04] Best G: 96.2% | C: 54.6% | Res: Male/Black
> [ID 05] Best G: 80.5% | C: 40.3% | Res: Female/White
> [ID 06] Best G: 98.1% | C: 35.0% | Res: Female/Gray
> [ID 07] Best G: 98.1% | C: 36.8% | Res: Female/Gray
> [ID 08] Best G: 74.8% | C: 49.4% | Res: Female/Blue
> [ID 10] Best G: 75.8% | C: 65.7% | Res: Male/Gray
> [ID 11] Best G: 98.9% | C: 56.7% | Res: Female/Gray
> [ID 12] Best G: 76.0% | C: 30.0% | Res: Female/Gray
> [ID 13] Best G: 85.6% | C: 66.5% | Res: Female/White
> [ID 14] Best G: 96.8% | C: 69.6% | Res: Female/Gray
> [ID 16] Best G: 97.9% | C: 72.9% | Res: Female/Gray
> [ID 17] Best G: 86.4% | C: 98.1% | Res: Female/Black
> [ID 18] Best G: 99.6% | C: 35.6% | Res: Male/Gray
> [ID 19] Best G: 83.1% | C: 87.6% | Res: Male/Gray
> [ID 20] Best G: 72.2% | C: 46.6% | Res: Female/Gray
> [ID 22] Best G: 98.3% | C: 40.7% | Res: Male/Blue

```
> **분석:** 주간 환경에서는 높은 Mean Confidence(87.9%)를 보이며 안정적인 속성 추출이 가능함을 확인했습니다. 특히 `ID 03`, `ID 18` 등에서 96% 이상의 높은 성별 판별력을 보여 VLM의 우수한 제로샷 성능을 확보했습니다.


- eval_night(야간실외 / 고정카메라) :
```bash
===================================================================================================================
[FINAL SYSTEM EVALUATION REPORT]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : eval_night.mp4
    - Processed Date : 2026-01-16 09:37:21
    - Storage Path   : logs/report_eval_night_20260116_093721.txt

 2. ATTRIBUTE DISTRIBUTION SUMMARY
    - Final Count (ROI)    : 22
    - Gender Distribution  : Male(15) / Female(17) / Unk(0)
    - Mean Conf (Gender)   : 76.56% (Analysis Reliability)

 3. SYSTEM PERFORMANCE
    - Effective FPS      : 16.1 (Target 15+ on 4070Ti)
    - Defense Strategy   : Aspect Ratio Filter(>1.5) + Best-Frame Sampling(3-step)
===================================================================================================================

[APPENDIX: INDIVIDUAL OBJECT LOGS]
> [ID 01] Best G: 65.9% | C: 50.1% | Res: Male/Gray
> [ID 02] Best G: 79.3% | C: 66.4% | Res: Male/Gray
> [ID 03] Best G: 64.7% | C: 81.3% | Res: Female/Black
> [ID 04] Best G: 83.2% | C: 58.4% | Res: Male/White
> [ID 09] Best G: 68.3% | C: 53.6% | Res: Female/Gray
> [ID 10] Best G: 61.5% | C: 68.8% | Res: Male/White
> [ID 11] Best G: 58.9% | C: 76.2% | Res: Female/Black
> [ID 12] Best G: 93.2% | C: 49.4% | Res: Male/Gray
> [ID 13] Best G: 91.1% | C: 50.8% | Res: Male/Red
> [ID 14] Best G: 81.8% | C: 48.0% | Res: Male/Black
> [ID 15] Best G: 73.9% | C: 58.7% | Res: Female/Gray
> [ID 16] Best G: 80.9% | C: 67.3% | Res: Male/White
> [ID 17] Best G: 83.9% | C: 73.1% | Res: Male/White
> [ID 18] Best G: 60.3% | C: 65.5% | Res: Male/Black
> [ID 19] Best G: 68.4% | C: 55.6% | Res: Female/Gray
> [ID 20] Best G: 95.4% | C: 34.9% | Res: Female/Black
> [ID 21] Best G: 84.5% | C: 37.2% | Res: Female/Black
> [ID 22] Best G: 93.9% | C: 26.7% | Res: Female/White
> [ID 25] Best G: 62.7% | C: 67.8% | Res: Male/Gray
> [ID 26] Best G: 87.7% | C: 41.6% | Res: Male/Gray
> [ID 27] Best G: 75.8% | C: 36.7% | Res: Female/Gray
> [ID 29] Best G: 78.6% | C: 89.3% | Res: Female/Red
> [ID 31] Best G: 65.4% | C: 40.6% | Res: Female/Black
> [ID 32] Best G: 60.2% | C: 63.6% | Res: Male/Black
> [ID 33] Best G: 76.0% | C: 30.6% | Res: Male/Black
> [ID 36] Best G: 94.2% | C: 40.5% | Res: Female/Blue
> [ID 37] Best G: 77.8% | C: 46.2% | Res: Female/Black
> [ID 38] Best G: 72.0% | C: 82.8% | Res: Female/White
> [ID 39] Best G: 93.1% | C: 74.9% | Res: Male/White
> [ID 40] Best G: 73.8% | C: 37.3% | Res: Female/Gray
> [ID 41] Best G: 75.5% | C: 53.2% | Res: Female/White
> [ID 42] Best G: 67.9% | C: 34.0% | Res: Female/White
```

>**분석:** 야간의 빛의 환경에서도 88.0%의 높은 정확도를 유지했습니다. 이는 시각적 노이즈를 극복하기 위해  **Aspect Ratio Filter**와 **Best-Frame Sampling** 적용한것에 대한 성능방어라고 판단했습니다.

- eval_indoors(실내/ 동적카메라) :
```bash

===================================================================================================================
[FINAL SYSTEM EVALUATION REPORT]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : eval_indoors.mp4
    - Processed Date : 2026-01-16 09:35:31
    - Storage Path   : logs/report_eval_indoors_20260116_093531.txt

 2. ATTRIBUTE DISTRIBUTION SUMMARY
    - Final Count (ROI)    : 30
    - Gender Distribution  : Male(19) / Female(14) / Unk(0)
    - Mean Conf (Gender)   : 85.06% (Analysis Reliability)

 3. SYSTEM PERFORMANCE
    - Effective FPS      : 15.5 (Target 15+ on 4070Ti)
    - Defense Strategy   : Aspect Ratio Filter(>1.5) + Best-Frame Sampling(3-step)
===================================================================================================================

[APPENDIX: INDIVIDUAL OBJECT LOGS]
> [ID 01] Best G: 86.8% | C: 68.9% | Res: Male/Black
> [ID 02] Best G: 69.1% | C: 47.2% | Res: Male/Black
> [ID 03] Best G: 97.6% | C: 87.0% | Res: Male/Red
> [ID 05] Best G: 93.4% | C: 79.4% | Res: Male/White
> [ID 07] Best G: 55.7% | C: 72.4% | Res: Female/Black
> [ID 08] Best G: 96.2% | C: 39.5% | Res: Female/Black
> [ID 09] Best G: 68.2% | C: 95.4% | Res: Female/Black
> [ID 10] Best G: 79.6% | C: 47.2% | Res: Male/Gray
> [ID 11] Best G: 91.0% | C: 31.1% | Res: Male/Gray
> [ID 12] Best G: 84.2% | C: 71.2% | Res: Male/Yellow
> [ID 13] Best G: 89.9% | C: 33.8% | Res: Male/Blue
> [ID 14] Best G: 95.0% | C: 56.0% | Res: Male/Gray
> [ID 15] Best G: 74.4% | C: 98.6% | Res: Female/Black
> [ID 16] Best G: 84.8% | C: 82.2% | Res: Female/Black
> [ID 17] Best G: 98.4% | C: 41.3% | Res: Male/Gray
> [ID 18] Best G: 96.4% | C: 54.2% | Res: Female/Gray
> [ID 21] Best G: 67.6% | C: 67.0% | Res: Female/White
> [ID 22] Best G: 94.3% | C: 31.7% | Res: Male/Gray
> [ID 24] Best G: 74.8% | C: 64.6% | Res: Female/Red
> [ID 25] Best G: 61.9% | C: 77.6% | Res: Female/Black
> [ID 26] Best G: 81.9% | C: 62.6% | Res: Female/Gray
> [ID 27] Best G: 95.4% | C: 56.0% | Res: Male/White
> [ID 28] Best G: 95.8% | C: 74.2% | Res: Male/Blue
> [ID 29] Best G: 79.8% | C: 41.5% | Res: Male/Gray
> [ID 30] Best G: 56.4% | C: 70.2% | Res: Female/Black
> [ID 31] Best G: 91.7% | C: 38.6% | Res: Male/Gray
> [ID 32] Best G: 93.4% | C: 57.8% | Res: Male/Blue
> [ID 33] Best G: 96.9% | C: 57.3% | Res: Male/Blue
> [ID 34] Best G: 97.1% | C: 72.6% | Res: Male/Red
> [ID 35] Best G: 97.0% | C: 39.3% | Res: Male/Black
> [ID 36] Best G: 98.3% | C: 76.7% | Res: Female/White
> [ID 37] Best G: 66.4% | C: 84.5% | Res: Female/Black
> [ID 38] Best G: 98.6% | C: 82.8% | Res: Female/Blue
```

>  **분석:** 카메라가 움직이는 동적 환경임에도 불구하고 ByteTrack과의 결합을 통해 30명의 인원을 성공적으로 트래킹했습니다. 배경 변화가 심한 실내에서도 평균 85%의 신뢰도를 유지하며 시스템의 범용성을 확보했습니다.


Yolov8n + SuperVision + Yolov8n-cls + HSV + K-Means Clustering
dev_day(주간실외 / 고정카메라) :
```bash
**
===================================================================================================================
[FINAL HYBRID SYSTEM REPORT (VOTING ENABLED)]
===================================================================================================================
 1. SESSION INFO
    - Target Video   : dev_day.mp4
    - Processed Date : 2026-01-16 09:34:16
    - Storage Path   : logs/report_LOW_dev_day_20260116_093416.txt

 2. PERFORMANCE SUMMARY
    - Total Tracked  : 35
    - Final Count    : 27
    - Gender Stats   : Male(20) / Female(5) / Unk(1)
    - Effective FPS  : 16.2
    
 3. CONFIGURATION
    - Mode           : Low-Spec / Hybrid Voting
    - Logic          : 3-Frame Temporal Voting (Majority Rule)
===================================================================================================================

[APPENDIX: INDIVIDUAL LOGS]
> [ID 01] Final: Male(27) | Green
> [ID 02] Final: Male(20) | Green
> [ID 03] Final: Male(17) | Green
> [ID 04] Final: Male(55) | Gray
> [ID 05] Final: Male(26) | Green
> [ID 06] Final: Male(18) | Blue
> [ID 07] Final: Male(16) | White
> [ID 08] Final: Unk(10) | Green
> [ID 09] Final: Female(17) | White
> [ID 10] Final: Male(39) | Black
> [ID 11] Final: Male(38) | White
> [ID 12] Final: Male(35) | Gray
> [ID 15] Final: Male(36) | White
> [ID 16] Final: Female(58) | White
> [ID 17] Final: Male(24) | White
> [ID 18] Final: Male(25) | White
> [ID 21] Final: Male(18) | White
> [ID 22] Final: Male(29) | White
> [ID 23] Final: Male(38) | Gray
> [ID 24] Final: Male(38) | Green
> [ID 25] Final: Male(22) | White
> [ID 26] Final: Female(29) | Black
> [ID 30] Final: Female(46) | White
> [ID 31] Final: Male(17) | White
> [ID 32] Final: Male(32) | Red
> [ID 33] Final: Female(17) | Green
**
```


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

>저사양 환경에서는 수학적 통계 방식의 한계로 인해 정확도가 하락하나, 이를 보완하기 위해 VLM 기반의 고사양 모드로 바꾸었습니다.
>또한 성능 비교를 하며 진행했기때문에 두 환경 모두 배포하는것으로 결정했습니다.

❌ 초기 설정(특성활용에 휴리스틱방식)  : 
- Yolov8n + HSV + K-means
임베디드 혹은 저사양 하드웨어 환경을 고려하여 수학적 통계방식을 채택하였으나
특징(성별,의상색깔)에 인식률과 퍼포먼스가 저조하고 다양한 환경(광량, 노이즈, 구도)에서 범용성에 맞게 사용하기 어렵다고 판단하여 기술을 변경.

✅ 최종채택 (특성활용에 VLM채택) : 
- Yolov8n + SuperVision + VLM(CLIP:ViT-B/16)
과제 특성상  `SW의 우수성을 입증할 수 있어야 함 (정확성, 추론속도, 기능)` 과 5일한정 마감기간에 따른 판단으로, 모델에 특성을 따로 학습을 할수 없었기 때문에 성능과 효율을 타협하여 인식률을 우선시 하여 채택했습니다.


⚠ 단점 보완
최신 VLM SOTA 모델을 채택하여 임베디드 혹은 저사양 하드웨어 환경에서 작동이 어려울수 있으나 RTX 4070ti 12GB 기준 FPS15+ 로 퍼포먼스 최적화도 신경써서 개발했습니다.

### 9-3. 추후 도전과제 (Future Works)

1. **성능 최적화 및 에지 디바이스(Edge Device) 대응**
	+  향후 임베디드 및 저사양 환경에서의 구동을 위해 **VLM(CLIP: ViT-B/16)** 및 **Qwen** 등 대형 VLM 모델을 **ONNX** 또는 **NCNN** 포맷으로의 변환
	+  INT8/FP16 양자화 를 적용하여 정밀도를 유지하면서 추론속도를 올리는 최적화를 추가할것을 기대

2. **일관된 배포 환경 구성**
    - 대규모 배포나 서버 환경 확장을 위해서는 **Docker**를 활용한 컨테이너 기반의 일관성 있는 배포가 필요
    
3. **데이터 관리 및 EDA 활용**
    - SQLite 연동을 통한 누적 카운팅 또는 특성 데이터저장
    - 분석된 성별,복장 데이터를 대시보드화하여 실시간 통계 시각화 로직 구현

4. **UI/UX 개선**
	+ 개발 프로토타입 이므로 상용화및 실무 배포 하기 위해서 깔끔한 폰트와 UI/UX를 추가

5. **하드웨어 성능별 브랜치관리**
	+ 고성능,일반성능,경량화 버전별로 개발 브랜치 나눠서 고도화

6. **TensorRT/ONNX 가속 적용**
	-  Intel iGPU (내장 그래픽12GB) 노트북환경에서 경량화 모델 구동시 FPS가 8~10 수준으로 나왔기때문에 저사양하드웨어에서 추론속도를 위해 TensorRT/ONNX 가속을 적용하여 실시간성(15 FPS+)확보하는것으로 개선
---

