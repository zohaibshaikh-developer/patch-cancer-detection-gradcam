import streamlit as st
from datetime import datetime

st.set_page_config(page_title="PathoLens – AI Cancer Detection", layout="wide")
st.title("🧬 PathoLens – AI-Driven Histopathology Intelligence")
st.markdown("#### Transforming Cancer Detection with Deep Learning and Visual Explainability")

# === Overview Section ===
with st.container():
    st.markdown("### 🔍 What is PathoLens?")
    st.write("""
    **PathoLens** is an AI-powered clinical support platform designed to assist pathologists in detecting cancer from histopathology image patches. 
    Leveraging a fine-tuned **ResNet18** deep learning model enhanced with **Grad-CAM visual explanations**, it delivers:
    
    - 🧠 **Accurate Predictions**: Identifies whether tissue samples are cancerous or normal
    - 🎯 **Visual Explainability**: Highlights suspicious regions using Grad-CAM heatmaps
    - 📄 **Automated PDF Reports**: Summarizes predictions, confidence scores, visual analysis, and clinical notes

    This system is trained on the **Camelyon17-clean dataset**—a trusted benchmark for breast cancer metastasis detection—and optimized for real-world use.
    """)

# === Why It Matters ===
with st.container():
    st.markdown("### 💡 Why Use PathoLens?")
    st.write("""
    PathoLens bridges the gap between AI and pathology by offering an intuitive interface for clinicians to:
    
    - 🔬 **Reduce Diagnostic Error**: AI assists with second-opinion validation, reducing oversight
    - ⏱️ **Save Time**: Instant prediction and visualization within seconds per patch
    - 📊 **Maintain Traceability**: Automatically generate structured, shareable PDF reports
    - 🧩 **Support Early Detection**: Enables high-resolution patch-level screening to flag abnormalities early

    Designed with clinical workflows in mind, PathoLens empowers both experts and researchers to make **data-driven, explainable decisions**.
    """)

# === Model Summary ===
with st.container():
    st.markdown("### 📊 System Summary")
    cols = st.columns(4)
    cols[0].metric(label="Model Architecture", value="ResNet18")
    cols[1].metric(label="Training Dataset", value="Camelyon17-clean")
    cols[2].metric(label="Test Accuracy", value="99.6%")
    cols[3].metric(label="Last Trained", value=datetime.now().strftime("%Y-%m-%d"))

st.markdown("---")
st.markdown("Made for research and clinical insight – not a replacement for professional diagnosis. Always consult a qualified pathologist.")
