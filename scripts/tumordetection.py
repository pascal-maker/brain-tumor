import os
from ultralytics import YOLO
from roboflow import Roboflow

# Load the YOLOv8 model (recommended for training)
model = YOLO("weights/yolov8n.pt")  # Using a pretrained model

# Set up the Roboflow project and download the dataset
rf = Roboflow(api_key="lvJtfzYNSRQrGmcNtdUd")
project = rf.workspace("workspace-hyza2").project("brain-tumor-detection-6n4sk")
version = project.version(1)
dataset = version.download("yolov8")

# Verify the dataset directory
dataset_dir = dataset.location

# Define the path to your dataset configuration file
data_config_path = "brain-tumor.yaml"

# Create brain-tumor.yaml file content with the correct paths
yaml_content = f"""
train: {os.path.join(dataset_dir, 'train/images')}
val: {os.path.join(dataset_dir, 'valid/images')}

nc: 2  # Number of classes (e.g., tumor, no tumor)
names: ['tumor', 'no_tumor']  # Class names
"""

# Write the yaml content to brain-tumor.yaml
with open(data_config_path, "w") as file:
    file.write(yaml_content)

# Train the model
results = model.train(data=data_config_path, epochs=10, imgsz=640)
