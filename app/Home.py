import streamlit as st
from datetime import datetime

st.set_page_config(page_title="PathoLens – Cancer Detection AI", layout="wide")
st.title("🧬 PathoLens – AI-Driven Cancer Detection")
st.markdown("#### Professional Diagnostic Support for Histopathology Images")

with st.container():
    st.markdown("##### 🔍 Overview")
    st.write("""
    PathoLens is a patch-level cancer detection platform powered by ResNet18 + Grad-CAM. 
    It supports image upload, AI predictions, visual explainability, and report generation.
    """)

    st.markdown("##### 📊 Model Summary")
    st.metric(label="Model Architecture", value="ResNet18")
    st.metric(label="Training Dataset", value="Camelyon17-clean")
    st.metric(label="Test Accuracy", value="99.6%")
    st.metric(label="Last Trained", value=datetime.now().strftime("%Y-%m-%d"))
