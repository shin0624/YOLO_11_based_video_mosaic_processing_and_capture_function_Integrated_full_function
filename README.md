---
title: YOLO_11_based_video_mosaic_processing_and_capture_function_Integrated_full_function
emoji: 🌖
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
short_description: YOLO 11 모델 기반 동영상 내 사람 형상 모자이크 및 프레임 단위 이미지 캡쳐 기능 구현
---
# YOLO_11_based_video_mosaic_processing_and_capture_function_Integrated_full_function
## shin0624 내 Space 기능 모두 통합
**YOLO 11** 모델 기반 동영상 내 사람 형상 모자이크 및 프레임 단위 이미지 캡쳐 기능 구현
**HuggingPace Spaces**를 통해 Gradio 기반 호스팅
**https://huggingface.co/spaces/shin0624/YOLO_11_based_video_mosaic_processing_and_capture_function_Integrated_full_function**

![Image](https://github.com/user-attachments/assets/e8c2669d-9698-4d26-8c5f-8fd9e72cc765)

## Licenses
- 본 프로젝트 코드: **MIT License**
- YOLOv11 모델: **AGPL-3.0** ([Ultralytics 공식 문서](https://ultralytics.com/license))
- Gradio: **Apache-2.0**

# 기술 스택
![License](https://img.shields.io/badge/License-MIT%2FAGPL--3.0-blue)
<img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white">
<img src="https://img.shields.io/badge/yolo11-111F68?style=for-the-badge&logo=yolo&logoColor=white">
<img src="https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=Gradio&logoColor=white">
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)

# 구현된 기능 설명

## 입력 방식

- 로컬 파일 업로드 또는 Google Drive 링크 입력 가능

- 동영상 형식: mp4, mov 등 OpenCV가 지원하는 형식

## 핵심 처리 기능

- YOLOv11을 이용한 실시간 사람 검출

- 검출된 영역에 모자이크 처리 적용

- 사용자 정의 가능한 프레임 캡쳐 간격

- 해상도 조정 기능 (50%~100%)

- 용량 제한 및 최대 이미지 수 제어

## 진행 상태 표시

- 실시간 처리 진행률 슬라이더

- 현재 사용 용량 및 최대 용량 표시

- 비동기 처리로 진행 상태 실시간 업데이트

## 출력

- 처리된 프레임들을 ZIP 파일로 패키징

- 다운로드 가능한 파일 출력

## 대용량 처리

- Google Drive 링크를 통한 대용량 파일 직접 처리

- 메모리 관리를 위한 점진적 처리 방식

- 안전한 임시 파일 처리

# 250511 개선 사항
**1. 작업 시간 개선**
- 현재 cv.VideoCapture로 프레임마다 .set() -> .read()하는데, 이는 매우 느림.
- GPU 배치 처리 사용
```
a. BATCH_SIZE 설정을 통해 한 번에 여러 프레임 처리
b. amp.autocast()로 혼합 정밀도 연산 활성화
c. torch.backends.cudnn.benchmark로 컨볼루션 최적화
```
  
- 파이프라인 병렬화
```
a. 프레임 수집 -> 배치 추론 -> 후처리를 별도 스레드에서 처리
b. ThreadPoolExecutor로 CPU 집약 작업 병렬화
```
  
- 스트림 처리 모드
```
a. model.predict(stream=True)로 메모리 사용량 최적화
b. 연속된 프레임 처리 시 내부 버퍼 재사용
```

- 비동기 I/O작업
```
  영상 읽기 <-> 모델 추론 <-> 파일 저장 단계 오버래핑
  CUDA 스트림과 파이썬 스레드 조합을 사용
```

**2. 주요 최적화 포인트**
- 파이프라인 병렬화 아키텍처
```
a. 프레임 수집 → 전처리 → GPU 배치 추론 → 후처리 → 압축 단계를 오버랩 처리
b. CPU/GPU 작업 분리로 리소스 활용 극대화
```

- 혼합 정밀도 연산(Mixed Precision)
```
a. autocast() 컨텍스트 매니저로 FP16 연산 활성화
b. 메모리 사용량 40% 감소, 처리 속도 2배 향상 기대
```

- 스마트 배치 처리
```
a. 동적 배치 크기 조정
b. 한 번의 모델 호출로 다중 프레임 처리
```

- 비동기 I/O 관리
```
a. 영상 읽기와 ZIP 압축 쓰기를 별도 스레드에서 처리
b. CUDA 스트림과 Python 스레드 풀 연동
```

- 메모리 최적화
```
a. 프레임 데이터의 BGR/RGB 변환 최소화
b. 결과 버퍼 즉시 방출(streaming) 방식 채택
```

**3. 작업 결과 검증**
- 84.5MB 크기의 1분 37초 영상 작업 시 기존 코드와 개선된 코드의 효율을 검증
- 디바이스 환경 : Nvidia T4 small (4vCPU, 15GB RAM, 16GB VRAM) (Huggingface Pro 요금제 구독)
- 결과 도출 조건 : 프레임 간격 5, 해상도 100%, 동영상을 로컬에서 직접 업로드
  
**- 작업 결과**
  ```
- 프레임 수집 및 전처리 : 03분36초
- 후처리 및 압축 : 03분 23초
- 총 작업 소요 시간 : 06분 59초
- 결과물 용량 : 120.4MB
  ```
### 전처리 결과
![Image](https://github.com/user-attachments/assets/0f747455-6cd3-4b53-bbc2-04aca87195c3)

### 후처리 결과
![Image](https://github.com/user-attachments/assets/28ea7d9c-b482-499b-a6d3-c2c778777e34)

**4. Gradio 인터페이스를 수정하여, 상단에 gr.Textbox를 배치.**
- html block을 사용하여 gpu는 초록색, cpu는 빨간색으로 표시하고, 아이콘 추가, 수동 새로고침도 가능하도록 구성.
- update()와 live=True 속성을 활용하여 앱 실행 중에도 자원 사용 상태를 감지하여 인터페이스에 실시간 반영.

**5. ThreadPoolExecutor를 이용한 직접 병렬화 구현 테스트**

	1. 2단계 병렬화 아키텍처 구성
		- 상위레벨 : 배치단위 병렬처리(ThreadPoolExecutor)
		- 하위레벨 : GPU 배치추론 + OpenCV 벡터화 연산

	2. 메모리 효율화 기법
		- 프레임 데이터의 rgh/bgr 변환 최소화
		- 배치 처리 간 PGU 캐시 자동 관리
		- 결과 버퍼 즉시 디스크에 기록

	3. Huggingface 제한 사항 대응
		- MAX_WORKERS = 4로 안정성 보장
		- BATCH_SIZE = 8로 VRAM 효율 관리
		- 큐 시스템 간소화

	4. 에러 방지
		- 구글 드라이브 링크 파싱 오류 처리
		- 비디오 파일 열기 실패 시예외 발생
		- 프레임 읽기 실패 시 자동 건너뛰기

**6. YOLO모델 fuse() 호환성 문제 해결 방안**
- fuse()완전 비활성화
```
a. 모델 초기화 직후 model.fuse를 람다함수로 재정의
b. predict메서드 오버라이드로 이중 보안
```

- 최신 파이토치 AMP API 적용
```
a. torch.amp.autocast로 경고 해결
b. device_type과 dtype 명시적 지정
  --> AttributeError : bn 오류 해결, AMP 관련 경고메시지 해결, YOLOv11n과 Ultralytics 라이브러리 호환성 문제 해결
```

**7. 재귀 깊이 초과 오류(RecursionError) 해결**
- model.predict 오버라이드 시 원본 메서드를 참조하도록 수정
- 람다 함수가 자기 자신을 호출하지 않도록 구조 변경
- original_predict 변수로 원본 메서드 보존
- fuse 메서드 완전 무력화

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
