import os
import csv
import torch
import numpy as np
import cv2
import glob
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# Paths

# Automatically get the latest misclassified_*.csv
misclassified_files = sorted(glob.glob("logs/misclassified_*.csv"))
if not misclassified_files:
    raise FileNotFoundError("No misclassified_*.csv file found in logs/")
misclassified_csv = misclassified_files[-1]
print(f"📂 Loaded misclassified samples from: {misclassified_csv}")

output_dir = "gradcam_misclassified"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/resnet18_epoch10_20250615_024251.pth"
model = models.resnet18(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

target_layers = [model.layer4[-1]]
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.7, 0.7, 0.7], [0.25, 0.25, 0.25])
])

# Read misclassified samples
with open(misclassified_csv, "r") as f:
    reader = csv.DictReader(f)
    samples = list(reader)

for row in tqdm(samples, desc="🔥 Grad-CAM++ Misclassified"):
    img_path = row["filename"]
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=img_tensor)[0, :]

    rgb_img = np.array(img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fname = os.path.basename(img_path)
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

print(f"✅ Saved Grad-CAM++ overlays to: {output_dir}")
