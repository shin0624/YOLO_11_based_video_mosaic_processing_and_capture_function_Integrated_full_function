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
from zipfile import ZipFile, ZIP_DEFLATED
import gdown
from tqdm import tqdm
import torch

# GPU 및 환경 설정
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 로드
model = YOLO('yolo11n.pt').to(device)
temp_dir = tempfile.mkdtemp()

def apply_mosaic(frame, x1, y1, x2, y2, mosaic_size=15):
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (mosaic_size, mosaic_size))
    roi = cv2.resize(roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = roi
    return frame

def process_video(
    video_source,
    drive_link,
    frame_interval=1.17,
    target_resolution=70,
    compression_level=3
):
    # 비디오 파일 경로 설정
    if drive_link:
        video_path = os.path.join(temp_dir, 'input.mp4')
        try:
            file_id = drive_link.split("/d/")[1].split("/")[0]
            direct_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(direct_url, video_path, quiet=False)
        except Exception as e:
            raise RuntimeError("Google Drive 링크 형식이 잘못되었습니다.") from e
    elif video_source:
        video_path = video_source.name  # 업로드된 비디오 파일 경로
    else:
        raise RuntimeError("업로드된 동영상이나 Google Drive 링크 중 하나가 필요합니다.")

    # 비디오 캡처
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("비디오를 열 수 없습니다.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = int(width * (target_resolution / 100))
    new_height = int(height * (target_resolution / 100))

    zip_path = os.path.join(temp_dir, 'output.zip')
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]

    with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zipf:
        for frame_idx in tqdm(range(0, total_frames, int(frame_interval)), desc="Processing"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"프레임 {frame_idx} 읽기 실패")
                continue

            results = model(frame, device=device)
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                frame = apply_mosaic(frame, x1, y1, x2, y2)

            if target_resolution != 100:
                frame = cv2.resize(frame, (new_width, new_height))

            _, buffer = cv2.imencode('.png', frame, png_params)
            zipf.writestr(f"frame_{frame_idx}.png", buffer)

    cap.release()
    return zip_path

# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("## 동영상 모자이크 처리 (GPU 가속)")

    with gr.Row():
        video_input = gr.Video(label="업로드 동영상")
        drive_input = gr.Textbox(label="Google Drive 링크")

    with gr.Column():
        frame_interval = gr.Slider(1.0, 5.0, value=1.17, step=0.01, label="프레임 간격")
        target_resolution = gr.Slider(50, 100, value=70, label="해상도 (%)")
        output_file = gr.File(label="결과 ZIP 파일")

    btn = gr.Button("처리 시작")
    btn.click(
        process_video,
        inputs=[video_input, drive_input, frame_interval, target_resolution],
        outputs=output_file
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860
    )
