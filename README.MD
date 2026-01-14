# Attribute-based People Counter with ReID (Single Camera)

## Project Overview

본 프로젝트는 **단일 카메라 환경에서 특정 속성(성별, 옷 색상 등)을 기준으로 사람을 선별 카운팅하는 AI 기반 People Counter 프로토타입**입니다.  
객체 탐지, 추적(ID 유지), 속성 추론, ROI 기반 카운팅까지의 **전체 파이프라인을 실무 적용 관점에서 설계**하는 것을 목표로 합니다.

본 과제는 단순한 데모 구현이 아닌,  
**실제 현장(Edge / Local 환경)에서 바로 확장·적용 가능한 구조**를 중점적으로 고려하여 진행하였습니다.

---

## Demo

- 📹 Demo Video (Counting Result Visualization)  
  👉 *(추후 업로드 예정)*  
  - Bounding Box + Tracking ID  
  - ROI 기반 선별 카운팅  
  - 속성별 카운트 오버레이 표시

---

## Table of Contents

1. Project Motivation & Design Intent  
2. System Architecture  
3. Dataset Description  
4. Attribute Definition  
5. Installation  
6. Execution  
7. Evaluation Strategy  
8. Project Structure  
9. Limitations & Future Work

---

## 1. Project Motivation & Design Intent

본 과제는 다음과 같은 실무 요구사항을 가정하고 설계되었습니다.

- **단일 카메라 환경**
- **사람 객체의 안정적인 추적(ID 유지)**
- **특정 속성 기준 선별 카운팅**
- **엣지 환경(Raspberry Pi 등)까지 고려 가능한 경량 구조**
- **README 하나만으로 빠르게 이해 및 실행 가능**

특히, GUI 중심의 데모보다는  
**AI 파이프라인의 명확성, 모듈화, 재현성**을 우선시하였습니다.

---

## 2. System Architecture

전체 파이프라인은 아래와 같은 흐름으로 구성됩니다.

1. **Person Detection**
   - YOLO 기반 사람 객체 탐지
2. **Tracking**
   - 프레임 간 ID 유지 (Multi-Object Tracking)
3. **Attribute Inference**
   - Bounding box 기반 속성(성별, 옷 색상 등) 추론
4. **ROI Filtering**
   - 미리 정의된 관심 영역(ROI) 내 객체만 처리
5. **Line / Area-based Counting**
   - 중복 카운트 방지 로직 포함
6. **Visualization**
   - 결과 오버레이 및 카운트 출력

---

## 3. Dataset Description

### 3.1 Original Dataset

본 프로젝트에서는 **MOT Challenge 2016 (MOT16)** 데이터셋을 사용하였습니다.

- MOT16-09  
  - Daytime pedestrian street scene  
  - Low-angle view  
  - High pedestrian density  
- MOT16-04  
  - Nighttime pedestrian street scene  
  - Elevated viewpoint  
  - Challenging illumination conditions  

### 3.2 Video Selection Rationale

해당 시퀀스는 다음 이유로 선정되었습니다.

- **다중 객체 환경에서의 추적 안정성 검증 가능**
- **조명 변화(주간 / 야간)에 따른 속성 추론 한계 확인**
- **실제 현장 환경과 유사한 혼잡도**

### 3.3 File Naming Convention

원본 데이터셋과의 명확한 대응 관계를 위해 아래와 같이 파일명을 변경하여 사용합니다.

| Original Sequence | Used Filename |
|------------------|--------------|
| MOT16-09 | `dev_day.mp4` |
| MOT16-04 | `eval_night.mp4` |

> ⚠️ 원본 영상은 저장소에 포함되어 있지 않습니다.

### 3.4 Dataset Download (Temporary)

아래 링크를 통해 영상을 다운로드한 후 `data/` 디렉토리에 배치해 주세요.

- `dev_day.mp4` : *(Temporary Link - To be updated)*  
- `eval_night.mp4` : *(Temporary Link - To be updated)*  

> 추후 Google Drive / Mega 링크로 교체 예정

---

## 4. Attribute Definition

본 프로젝트에서는 **명확하게 정의 가능한 속성**을 중심으로 설계하였습니다.

- **Clothing Color**
  - 상체 영역 기준
  - HSV 색공간 기반 단순화된 색상 그룹 분류
- **Gender (Heuristic-based)**
  - Bounding box 비율, 상·하체 비율 기반 추정
  - 연구 목적이 아닌 **현장 적용 가능성 검증용**

> 모든 속성 추론 결과는 “추정값”이며,  
> 정확도보다 **일관성과 재현성**을 우선합니다.

---

## 5. Installation

### Environment

- Python 3.9+
- OS: Linux / Windows
- GPU: Optional (CUDA 지원 시 자동 사용)

### Dependencies

```bash
pip install -r requirements.txt
```