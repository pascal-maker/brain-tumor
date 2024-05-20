import os
import glob
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a brain-tumor fine-tuned model

# Construct the file pattern using os.path.join
base_path = "assets/brain_tumor_dataset/yes"
file_pattern = "*.jpg"
image_path = os.path.join(base_path, file_pattern)

# Find files matching the pattern
image_files = glob.glob(image_path)

# Verify the working directory
current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")

# Check if any image files were found
if image_files:
    # Inference using the model
    results = model.predict(image_files[0])
    print(results)
else:
    print("No image files found in the specified directory.")
