import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO
import os
from PIL import Image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def find_target_layer(model):
    # For ResNet18, last convolutional layer is typically layer4[-1]
    return model.layer4[-1]

def apply_gradcam_and_interpret(model, image, target_layer, class_idx):
    original_np = np.array(original_pil.resize((224, 224))).astype(np.float32) / 255.0
    original_np = original_np[..., :3]  # remove alpha if any

    # Initialize GradCAM
    target_layer = find_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=None)
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=None)[0]  # HWC, float32

    # === Colored heatmap ===
    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # === Overlay ===
    overlay_np = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)

    # Convert all to PIL
    heatmap_img = Image.fromarray(heatmap_rgb)
    overlay_img = Image.fromarray(overlay_np)
    original_img = original_pil.resize((224, 224))

    # === Interpretation Logic ===
    focus_ratio = (grayscale_cam > 0.5).mean()
    coords = np.column_stack(np.where(grayscale_cam > 0.5))
    center_distance = 1.0
    if coords.any():
        y, x = coords.mean(axis=0)
        h, w = grayscale_cam.shape
        norm_x, norm_y = x / w, y / h
        center_distance = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)

    # === Clinical Observation Rules ===
    if focus_ratio > 0.3 and center_distance < 0.3:
        observation = "High central attention on dense region—suggestive of carcinoma focus."
        clinical_note = "Considered likely malignant. Follow-up histopathological confirmation advised."
    elif focus_ratio < 0.1:
        observation = "Diffuse attention with low model certainty."
        clinical_note = "Inconclusive focus; could represent ambiguous tissue. Recommend retesting."
    elif focus_ratio >= 0.1 and center_distance >= 0.6:
        observation = "Peripheral activation—possibly boundary tissue or well-differentiated zone."
        clinical_note = "No critical concern. Likely benign."
    else:
        observation = "Moderate attention around tissue clusters."
        clinical_note = "Likely benign, but attention map suggests areas worth inspection."

    return original_img, heatmap_img, overlay_img, observation, clinical_note, (norm_x, norm_y), focus_ratio

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
