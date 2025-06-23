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

def apply_gradcam_and_interpret(model, image, target_layer, class_idx, device):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import numpy as np

    # Prepare image for Grad-CAM
    image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    input_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    heatmap = (grayscale_cam * 255).astype(np.uint8)
    overlay = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    # Interpret results
    focus_score = (grayscale_cam > 0.5).mean()
    clinical_note = (
        "⚠️ High attention on abnormal regions. Immediate review advised."
        if focus_score > 0.3 else
        "✅ Attention is low. Likely benign, but clinical context is still required."
    )

    return Image.fromarray(overlay), Image.fromarray(heatmap), clinical_note, focus_score


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
