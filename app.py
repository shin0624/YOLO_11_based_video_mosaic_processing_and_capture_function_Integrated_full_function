# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 [shin0624]
# YOLOv11 부분은 AGPL-3.0 라이선스 적용
import os
import cv2
import numpy as np
import tempfile
import gradio as gr
from ultralytics import YOLO
from zipfile import ZipFile
import gdown
import time
from tqdm import tqdm

# YOLO 설정 디렉토리 문제 해결
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

# 모델 로드 (yolov8n.pt로 수정)
model = YOLO('yolo11n.pt')
temp_dir = tempfile.mkdtemp()

def apply_mosaic(frame, x, y, w, h, mosaic_size=15):
    area = frame[y:y+h, x:x+w]
    area = cv2.resize(area, (mosaic_size, mosaic_size))
    area = cv2.resize(area, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = area
    return frame

def process_video(
    video_source,
    drive_link,
    frame_interval,
    max_capacity,
    max_images,
    target_resolution,
    progress=gr.Progress()
):
    # Google Drive에서 파일 다운로드
    if drive_link:
        output = os.path.join(temp_dir, 'downloaded_video.mp4')
        gdown.download(drive_link, output, quiet=False)
        video_path = output
    else:
        video_path = video_source

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    zip_path = os.path.join(temp_dir, 'captured_frames.zip')
    current_capacity = 0
    captured_count = 0
    
    with ZipFile(zip_path, 'w') as zipf:
        for frame_idx in progress.tqdm(range(total_frames), desc="프레임 처리 중"):
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 처리
            if frame_idx % frame_interval == 0:
                results = model(frame)
                
                # 사람 검출 및 모자이크 적용
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, cls in zip(boxes, classes):
                        if cls == 0:  # 0은 사람 클래스
                            x1, y1, x2, y2 = map(int, box)
                            frame = apply_mosaic(frame, x1, y1, x2-x1, y2-y1)

                # 해상도 조정
                if target_resolution != 100:
                    h, w = frame.shape[:2]
                    new_w = int(w * (target_resolution / 100))
                    new_h = int(h * (target_resolution / 100))
                    frame = cv2.resize(frame, (new_w, new_h))

                # 이미지 저장
                if captured_count < max_images:
                    img_name = f"frame_{frame_idx}.jpg"
                    ret, buffer = cv2.imencode('.jpg', frame)
                    
                    img_size = len(buffer.tobytes())
                    if current_capacity + img_size > max_capacity * 1024 * 1024:
                        break
                    
                    zipf.writestr(img_name, buffer.tobytes())
                    current_capacity += img_size
                    captured_count += 1

    cap.release()
    return zip_path

with gr.Blocks() as demo:
    gr.Markdown("## 동영상 모자이크 처리 및 프레임 캡쳐")
    
    with gr.Row():
        video_input = gr.Video(label="업로드 동영상", sources=["upload"])
        drive_input = gr.Textbox(label="Google Drive 링크 (선택사항)")
    
    with gr.Row():
        with gr.Column():
            frame_interval = gr.Slider(1, 60, value=10, label="프레임 간격")
            target_resolution = gr.Slider(50, 100, value=100, step=5, label="해상도 (%)")
            max_capacity = gr.Number(100, label="최대 용량 제한 (MB)", precision=0)
            max_images = gr.Number(50, label="최대 이미지 수", precision=0)
            
        with gr.Column():
            progress_slider = gr.Slider(0, 100, value=0, label="진행 상태", interactive=False)
            capacity_info = gr.Textbox(label="용량 정보", value="0MB / 100MB")
            output_file = gr.File(label="다운로드 ZIP 파일")

    btn = gr.Button("처리 시작")
    
    def update_capacity(current, max_cap):
        return f"{current/(1024*1024):.2f}MB / {max_cap}MB"
    
    btn.click(
        process_video,
        inputs=[video_input, drive_input, frame_interval, max_capacity, max_images, target_resolution],
        outputs=output_file
    )

if __name__ == "__main__":
    demo.queue().launch()