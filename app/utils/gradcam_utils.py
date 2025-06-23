import numpy as np
import cv2
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO
from PIL import Image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def find_target_layer(model):
    return model.layer4[-1]  # ResNet18's last conv layer
def apply_gradcam_and_interpret(model, image, target_layer, class_idx, device):
    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Initialize GradCAM without deprecated use_cuda
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]  # shape: (H, W)

    # Normalize original image to [0,1]
    rgb_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    rgb_np = rgb_np[..., :3]  # Ensure 3 channels

    # Create heatmap using Jet colormap
    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay Grad-CAM heatmap on image
    overlay_np = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    # Convert to PIL
    heatmap_img = Image.fromarray(heatmap_rgb)
    overlay_img = Image.fromarray(overlay_np)
    original_img = image.resize((224, 224))

    # === Attention Interpretation ===
    focus_ratio = float((grayscale_cam > 0.5).mean())  # % of patch model focused on

    # Calculate center of attention
    coords = np.column_stack(np.where(grayscale_cam > 0.5))
    if coords.size > 0:
        y, x = coords.mean(axis=0)
        h, w = grayscale_cam.shape
        norm_x, norm_y = x / w, y / h
    else:
        norm_x, norm_y = 0.5, 0.5

    # === Clinical Interpretation Rules ===
    if focus_ratio > 0.3 and np.sqrt((norm_x - 0.5) ** 2 + (norm_y - 0.5) ** 2) < 0.3:
        observation = "High central attention on dense region—suggestive of carcinoma focus."
        note = "⚠️ Likely malignant. Recommend histopathology confirmation."
    elif focus_ratio < 0.1:
        observation = "Diffuse attention with low confidence."
        note = "⚠️ Inconclusive result. Retesting recommended."
    elif focus_ratio >= 0.1 and np.sqrt((norm_x - 0.5) ** 2 + (norm_y - 0.5) ** 2) >= 0.6:
        observation = "Peripheral attention—possibly benign or transitional tissue."
        note = "✅ No immediate concern."
    else:
        observation = "Moderate attention on irregular tissue clusters."
        note = "⚠️ Review advisable."

    return overlay_img, heatmap_img, note, focus_ratio, (norm_x, norm_y), observation

def generate_pdf_report(original_pil, heat_pil, overlay_pil, class_idx, obs, note):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Patch-Level Cancer Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Prediction: {'Cancerous' if class_idx else 'Normal'}", styles["Heading2"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"AI Observation: {obs}", styles["Normal"]))
    elements.append(Paragraph(f"Clinical Note: {note}", styles["Normal"]))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Visual Analysis", styles["Heading2"]))

    for img, title in zip([original_pil, heat_pil, overlay_pil], ["Original", "Heatmap", "Overlay"]):
        temp_path = f"/tmp/{title}.png"
        img.save(temp_path)
        elements.append(Paragraph(title, styles["Heading3"]))
        elements.append(PDFImage(temp_path, width=2.5 * inch, height=2.5 * inch))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer
