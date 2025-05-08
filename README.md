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


## Licenses
- 본 프로젝트 코드: **MIT License**
- YOLOv11 모델: **AGPL-3.0** ([Ultralytics 공식 문서](https://ultralytics.com/license))
- Gradio: **Apache-2.0**

# 기술 스택
![License](https://img.shields.io/badge/License-MIT%2FAGPL--3.0-blue)
<img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white">
<img src="https://img.shields.io/badge/yolo11-111F68?style=for-the-badge&logo=yolo&logoColor=white">
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)

# 구현된 기능 설명

## 입력 방식

- 로컬 파일 업로드 또는 Google Drive 링크 입력 가능

- 동영상 형식: mp4, mov 등 OpenCV가 지원하는 형식

## 핵심 처리 기능

- YOLOv8을 이용한 실시간 사람 검출

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


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
