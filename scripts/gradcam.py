import os
import json
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from tqdm import tqdm
import cv2
from datetime import datetime

# === Paths and Settings ===
data_path = "dataset/binary_split/val"
model_path = "models/resnet18_epoch10_20250615_024251.pth"
output_dir = "gradcam_outputs"
original_dir = "gradcam_originals"
metadata_file = os.path.join(output_dir, "metadata.json")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(original_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = models.resnet18(weights=None, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Load Data ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.7, 0.7, 0.7], [0.25, 0.25, 0.25])
])
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.25] * 3),
    transforms.Normalize(mean=[-0.7] * 3, std=[1.] * 3),
    transforms.ToPILImage()
])
dataset = ImageFolder(data_path, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# === Setup GradCAM++ ===
target_layers = [model.layer4[-1]]
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

metadata = {}

# === Process N images ===
NUM_SAMPLES = 10
processed = 0
for i, (images, labels) in enumerate(tqdm(loader, desc="Generating Grad-CAM", ncols=100)):
    if processed >= NUM_SAMPLES:
        break
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    pred = torch.argmax(outputs, dim=1)

    # Generate Grad-CAM++
    grayscale_cam = cam(input_tensor=images)[0]
    heatmap = np.uint8(255 * grayscale_cam)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Reverse normalization for original image
    image_np = inv_transform(images[0].cpu())
    image_np = np.array(image_np)

    # Resize and overlay
    heatmap_colored = cv2.resize(heatmap_colored, (image_np.shape[1], image_np.shape[0]))
    overlay = cv2.addWeighted(image_np, 0.5, heatmap_colored, 0.5, 0)

    # File naming
    fname = f"{'incorrect' if pred != labels else 'correct'}_{processed}_pred{pred.item()}_true{labels.item()}.png"
    overlay_path = os.path.join(output_dir, fname)
    orig_path = os.path.join(original_dir, fname)

    # Save images
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    Image.fromarray(image_np).save(orig_path)

    # === Analyze heatmap for metadata ===
    heatmap_bin = (grayscale_cam > 0.4).astype(np.uint8)
    focus_area = np.sum(heatmap_bin)
    total_area = heatmap_bin.size
    focus_ratio = round(focus_area / total_area, 4)

    # Compute center of mass
    h, w = grayscale_cam.shape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    total_activation = np.sum(grayscale_cam)
    if total_activation == 0:
        cx, cy = w // 2, h // 2
    else:
        cx = np.sum(x_grid * grayscale_cam) / total_activation
        cy = np.sum(y_grid * grayscale_cam) / total_activation

    dist_center = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
    max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    center_distance = round(dist_center / max_dist, 4)

    # Store metadata
    metadata[fname] = {
        "true_label": int(labels.item()),
        "pred_label": int(pred.item()),
        "focus_ratio": focus_ratio,
        "center_distance": center_distance
    }

    processed += 1

# === Save metadata ===
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n✅ Saved Grad-CAM overlays to: {output_dir}")
print(f"🧠 Saved original images to: {original_dir}")
print(f"📊 Saved metadata to: {metadata_file}")
