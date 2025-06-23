# report_generator_detailed.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import os
import json
import csv

# === Paths ===
gradcam_dir = "gradcam_outputs"
original_dir = "gradcam_originals"
metadata_path = os.path.join(gradcam_dir, "metadata.json")
metrics_csv = sorted([f for f in os.listdir("logs") if f.startswith("test_metrics")])[-1]
pdf_path = "Patch_GradCAM_Report_Professional.pdf"

# === Load metadata ===
with open(os.path.join(gradcam_dir, "metadata.json"), "r") as f:
    metadata = json.load(f)

# === Setup PDF ===
doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=30)
styles = getSampleStyleSheet()
elements = []

# === Cover Page ===
elements.append(Paragraph("<b>Patch-Level Cancer Detection Using ResNet18 and Grad-CAM</b>", styles["Title"]))
elements.append(Paragraph("AI Research & Clinical Interpretation Report", styles["Heading2"]))
elements.append(Spacer(1, 12))
elements.append(Paragraph("Author: <b>Zohaib Shaikh</b>", styles["Normal"]))
elements.append(Paragraph("Role: <b>AI/ML Engineer</b>", styles["Normal"]))
elements.append(Paragraph("Organization: <b>McAttoh</b>", styles["Normal"]))
elements.append(Paragraph(f"Generated on: <b>{datetime.now().strftime('%Y-%m-%d')}</b>", styles["Normal"]))
elements.append(PageBreak())

# === Executive Summary ===
elements.append(Paragraph("<b>1. Executive Summary</b>", styles["Heading1"]))
elements.append(Paragraph(
    "This report presents the output of an AI-powered pipeline for patch-level cancer detection trained on the Camelyon17 dataset. "
    "Using Grad-CAM++, we visualize model attention and extract clinical interpretations to support diagnostic workflows.",
    styles["Normal"]))
elements.append(PageBreak())

# === Dataset & Methodology ===
elements.append(Paragraph("<b>2. Dataset & Methodology</b>", styles["Heading1"]))
elements.append(Paragraph(
    "We used Camelyon17-clean, a high-resolution histopathology dataset. Patches (224×224) were extracted from WSIs and labeled as cancerous or normal. "
    "The model was trained using mixed precision with Adam optimizer, 10 epochs, batch size of 32, and cross-entropy loss.",
    styles["Normal"]))
elements.append(PageBreak())

# === Model & Training Setup ===
elements.append(Paragraph("<b>3. Model Architecture</b>", styles["Heading1"]))
elements.append(Paragraph(
    "We use a standard ResNet18 CNN trained from scratch on the patch data. Grad-CAM++ is applied on the final convolutional block. "
    "Training used an NVIDIA A4000 GPU. Model achieved >95% test accuracy. Augmentations include flips, normalization, and rescaling.",
    styles["Normal"]))
elements.append(PageBreak())

# === Metrics ===
elements.append(Paragraph("<b>4. Evaluation Metrics</b>", styles["Heading1"]))
if os.path.exists(f"logs/{metrics_csv}"):
    with open(f"logs/{metrics_csv}", "r") as f:
        rows = [row for row in csv.reader(f) if row]
    table = Table(rows)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica')
    ]))
    elements.append(table)
elements.append(PageBreak())

# === Grad-CAM Results with Interpretation ===
elements.append(Paragraph("<b>5. Visual Interpretation & Clinical Insight</b>", styles["Heading1"]))
images = sorted([f for f in os.listdir(gradcam_dir) if f.endswith(".png")])[:6]  # Limit to 6 for space

for fname in images:
    overlay_path = os.path.join(gradcam_dir, fname)
    orig_path = os.path.join(original_dir, fname)
    if not os.path.exists(orig_path):
        continue

    meta = metadata.get(fname, {})
    ratio = meta.get("focus_ratio", 0)
    dist = meta.get("distance_to_center", 1.0)

    # Interpret
    if ratio > 0.3 and dist < 0.3:
        obs = "High central focus — typical for malignant core tissue."
        note = "Follow-up biopsy strongly recommended."
    elif ratio < 0.1:
        obs = "Weak attention; possibly misclassified or ambiguous sample."
        note = "Low certainty — Retesting advised."
    elif ratio > 0.1 and dist > 0.6:
        obs = "Peripheral focus — possibly benign or reactive zone."
        note = "No immediate concern; monitor."
    else:
        obs = "Cluster-level activation; mildly suspicious."
        note = "Review under microscope suggested."

    elements.append(Paragraph(f"<b>Patch:</b> {fname}", styles["Normal"]))
    row = [
        Image(orig_path, width=2.2 * inch, height=2.2 * inch),
        Image(overlay_path, width=2.2 * inch, height=2.2 * inch)
    ]
    elements.append(Table([row], colWidths=[2.3 * inch] * 2))
    elements.append(Paragraph(f"<b>AI Observation:</b> {obs}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Clinical Note:</b> {note}", styles["Normal"]))
    elements.append(Spacer(1, 12))

elements.append(PageBreak())

# === Conclusion ===
elements.append(Paragraph("<b>6. Conclusion & Future Work</b>", styles["Heading1"]))
elements.append(Paragraph(
    "Grad-CAM++ provides transparency into deep learning models for cancer detection. This tool supports pathologists with visual cues. "
    "Future improvements include WSI-level MIL, integration with EMR systems, and SHAP-based cross-validation with pathologist annotations.",
    styles["Normal"]))

# === Build PDF ===
doc.build(elements)
print(f"✅ PDF generated: {pdf_path}")
