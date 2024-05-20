from ultralytics import YOLO

# Load a model
model = YOLO("weights\yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="brain-tumor.yaml", epochs=10, imgsz=640)