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
from concurrent.futures import ThreadPoolExecutor
from torch.amp import autocast

# fuse 메서드 핵심 패치 (모델 로드 전에 실행)
from ultralytics.nn.tasks import BaseModel

def patched_fuse(self, verbose=False):
    return self

BaseModel.fuse = patched_fuse  # fuse 기능 완전 무력화

# 병렬 처리 상수
MAX_WORKERS = 4
BATCH_SIZE = 8

# GPU 최적화 설정
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 (패치 적용 후 로드)
model = YOLO('yolo11n.pt').to(device)
model.eval()

temp_dir = tempfile.mkdtemp()

def apply_mosaic(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2:
            continue
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (15, 15), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = cv2.resize(roi, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    return frame

def process_batch(batch):
    frame_indices, frames = zip(*batch)
    
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        results = model(frames, verbose=False)
    
    processed = []
    for idx, frame, result in zip(frame_indices, frames, results):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        boxes = result.boxes.xyxy.cpu().numpy()
        processed.append((idx, apply_mosaic(frame, boxes)))
    return processed

def process_video(video_source, drive_link, frame_interval=1.17, target_resolution=70, compression=3):
    video_path = get_video(video_source, drive_link)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 프레임 수집
    batches = []
    current_batch = []
    for idx in tqdm(range(0, total_frames, int(frame_interval)), desc="Collecting Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            current_batch.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(current_batch) >= BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []
    if current_batch:
        batches.append(current_batch)

    # 해상도 계산
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w = int(orig_w * target_resolution / 100)
    new_h = int(orig_h * target_resolution / 100)
    cap.release()

    # 병렬 처리
    zip_path = os.path.join(temp_dir, 'output.zip')
    with ZipFile(zip_path, 'w', ZIP_DEFLATED) as zipf, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        # 배치 처리 작업 생성
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        for future in tqdm(futures, desc="Processing Batches"):
            processed_frames = future.result()
            for idx, frame in processed_frames:
                if target_resolution != 100:
                    frame = cv2.resize(frame, (new_w, new_h))
                _, buffer = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, compression])
                zipf.writestr(f"frame_{idx}.png", buffer)

    return zip_path

def get_video(video_source, drive_link):
    if drive_link:
        file_id = drive_link.split("/d/")[1].split("/")[0]
        video_path = os.path.join(temp_dir, 'input.mp4')
        gdown.download(f"https://drive.google.com/uc?id={file_id}", video_path, quiet=False)
        return video_path
    return video_source

def get_device_status():
    if torch.cuda.is_available():
        return '<span style="color:green; font-weight:bold;">✅ GPU 가속 활성화</span>'
    return '<span style="color:red; font-weight:bold;">⚠️ CPU 모드</span>'

# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("## 🚀YOLO11 기반 동영상 모자이크 처리 시스템(GPU / CPU 자동 탐지)")
    gr.HTML(get_device_status())

    with gr.Row():
        video_input = gr.Video(label="동영상 업로드", sources=["upload"])
        drive_input = gr.Textbox(label="Google Drive 링크", placeholder="https://drive.google.com/...")

    with gr.Column():
        frame_interval = gr.Slider(1, 5, 1.17, step=0.1, label="프레임 추출 간격 (초)")
        target_resolution = gr.Slider(30, 100, 70, label="출력 해상도 (%)")
        compression = gr.Slider(0, 9, 3, step=1, label="PNG 압축 수준")

    output_file = gr.File(label="처리 결과")
    start_btn = gr.Button("▶️ 처리 시작", variant="primary")
    
    start_btn.click(
        process_video,
        inputs=[video_input, drive_input, frame_interval, target_resolution, compression],
        outputs=output_file
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
