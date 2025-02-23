import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set environment variable for MPS fallback (if applicable)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load SAM 2 model
sam2_checkpoint = "/content/segment-anything-2/checkpoints/sam2_hiera_large.pt"  # Update path as needed
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Function to process the image and generate segmentation
def segment_image(image, x_coord=None, y_coord=None):
    # Convert Gradio image (PIL) to numpy array
    image_np = np.array(image)

    # Set the image in the predictor
    with torch.inference_mode(), torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
        predictor.set_image(image_np)

        # If point coordinates are provided, use them as prompts
        if x_coord is not None and y_coord is not None and x_coord > 0 and y_coord > 0:
            point_coords = np.array([[x_coord, y_coord]])
            point_labels = np.array([1])  # 1 for positive point (foreground)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels
            )
        else:
            # Fallback to automatic mask generation if no points provided
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
            masks = mask_generator.generate(image_np)

    # Visualize the mask
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)

    # Overlay masks
    for mask in masks:
        if isinstance(mask, dict):  # For automatic mask generator
            m = mask['segmentation']
        else:  # For predictor with point prompts
            m = mask
        color_mask = np.concatenate([np.random.random(3), [0.6]])
        mask_image = m.reshape(m.shape[0], m.shape[1], 1) * color_mask.reshape(1, 1, -1)
        ax.imshow(mask_image)

    ax.axis('off')
    plt.tight_layout()

    # Save the plot to a temporary file and return it
    output_path = "segmented_image.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return output_path

# Gradio interface
interface = gr.Interface(
    fn=segment_image,
    inputs=[
        gr.Image(type="pil", label="Upload Medical Image"),
        gr.Number(label="X Coordinate (optional)", value=0),
        gr.Number(label="Y Coordinate (optional)", value=0)
    ],
    outputs=gr.Image(label="Segmented Image"),
    title="Medical Image Segmentation with SAM 2",
    description="Upload a medical image (e.g., MRI scan) to segment regions of interest. Optionally provide X, Y coordinates for a specific point prompt (e.g., tumor center). If no coordinates are provided, automatic segmentation will be performed."
)

# Launch the app
interface.launch()
