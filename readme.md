# 🧠 Patch-Level Cancer Detection with Grad-CAM (Camelyon17)

This Streamlit app uses a ResNet18 model trained on the Camelyon17-clean dataset to detect cancerous tissue in histopathology patches. It generates Grad-CAM heatmaps for visual explainability and produces professional PDF reports for clinical interpretation.

## 🚀 Features
- Upload PNG/JPG patch
- Run live ResNet18-based prediction
- Visualize Grad-CAM, overlay, and attention focus
- View confidence, observation, and AI-generated clinical note
- Download detailed PDF report with explanation

## 🛠 Tech Stack
- Python 3.9+
- PyTorch + TorchVision
- Streamlit
- ReportLab
- OpenCV

## 📦 Installation (Local)
```bash
git clone https://github.com/yourusername/cancer_detection_app.git
cd cancer_detection_app
pip install -r requirements.txt
streamlit run app/pages/1_Upload_and_Predict.py
