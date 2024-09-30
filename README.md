# Brain Tumor Segmentation with SAM2 and YOLOv8

This repository demonstrates how to perform brain tumor segmentation using the Segment Anything Model 2 (SAM2) and YOLOv8 for object detection.

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Training](#training)
4. [Inference](#inference)
5. [Video Segmentation](#video-segmentation)
6. [Results](#results)

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- Roboflow API Key
- FFmpeg (for video processing)

Install the required packages using pip:
```bash
pip install ultralytics roboflow torch torchvision matplotlib pillow opencv-python
Setup
Step 1: Clone the Repository

git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
Step 2: Install Dependencies

pip install -e .
pip install -e ".[demo]"
Step 3: Download Model Checkpoints

cd checkpoints
./download_ckpts.sh
cd ..
Step 4: Verify CUDA Availability

import torch
print(torch.cuda.is_available())
Training
The training process involves setting up the YOLOv8 model and preparing the dataset using Roboflow.

Load Pretrained YOLOv8 Model

from ultralytics import YOLO
model = YOLO("weights/yolov8n.pt")
Set Up Roboflow Project and Download Dataset

from roboflow import Roboflow
rf = Roboflow(api_key="lvJtfzYNSRQrGmcNtdUd")
project = rf.workspace("workspace-hyza2").project("brain-tumor-detection-6n4sk")
version = project.version(1)
dataset = version.download("yolov8")
Verify Dataset Directory

dataset_dir = dataset.location
Create Dataset Configuration File (brain-tumor.yaml)

import os

data_config_path = "brain-tumor.yaml"
yaml_content = f"""
train: {os.path.join(dataset_dir, 'train/images')}
val: {os.path.join(dataset_dir, 'valid/images')}

nc: 2  # Number of classes (e.g., tumor, no tumor)
names: ['tumor', 'no_tumor']  # Class names
"""

with open(data_config_path, "w") as file:
    file.write(yaml_content)
Train the Model

results = model.train(data=data_config_path, epochs=10, imgsz=640)
Inference
Perform inference using the trained YOLOv8 model and SAM2 for segmentation.

Load Trained Model

model = YOLO("runs/train/exp/weights/best.pt")
Load Image

from PIL import Image
image = Image.open('/content/Y10.jpg')
image = np.array(image.convert("RGB"))
Generate Masks with SAM2

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "/content/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
masks = mask_generator.generate(image)
Visualize Results

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
Video Segmentation
For video segmentation, the process involves running SAM2 across multiple frames and saving the results as a video.

Load Video Frames

video_dir = "/content/output_frames"
frame_names = sorted([p for p in os.listdir(video_dir) if p.endswith(".jpg")])
Propagate Segmentation Across Frames

inference_state = None  # Initialize inference state
video_segments = {}

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    inference_state["tracking_has_started"] = True
Save Segmentation Results as Video

def save_video(output_path, video_segments, frame_names, video_dir, vis_frame_stride=30):
    sample_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    frame_width, frame_height = sample_frame.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        frame = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                if out_mask.shape != (frame_height, frame_width):
                    out_mask = np.reshape(out_mask, (frame_height, frame_width))
                colored_mask = np.zeros_like(frame)
                colored_mask[out_mask] = [0, 255, 0]
                alpha = 0.5
                frame = cv2.addWeighted(colored_mask, alpha, frame, 1 - alpha, 0)

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at {output_path}")

output_video_path = "segmented_output.mp4"
save_video(output_video_path, video_segments, frame_names, video_dir)
Results
The final output will be a video file (segmented_output.mp4) where each frame has been segmented to highlight brain tumors. The segmentation results can also be visualized interactively within Jupyter Notebook.

By following these steps, you can effectively perform brain tumor segmentation using SAM2 and YOLOv8. ```

Download README.md
README
.md
5.4 KB ⇣

