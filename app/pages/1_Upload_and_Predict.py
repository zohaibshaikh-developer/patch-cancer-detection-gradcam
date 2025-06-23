import streamlit as st
from app.utils.model_loader import predict_and_visualize
from PIL import Image

st.set_page_config(page_title="Upload and Predict", layout="wide")
st.title("📤 Upload Histopathology Patch")

st.markdown("Upload a Camelyon17 patch image below to run AI-based cancer detection with Grad-CAM interpretation.")

uploaded = st.file_uploader("🖼 Upload a .png or .jpg patch", type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded, caption="🧩 Uploaded Patch", width=250)

    with st.spinner("🔍 Running model inference and Grad-CAM..."):
        result = predict_and_visualize(uploaded)

    # === Prediction and Confidence ===
    label = "✅ Normal Tissue" if result["pred"] == 0 else "❗ Abnormal / Cancerous"
    color = "green" if result["pred"] == 0 else "red"
    st.markdown(f"### 🧠 Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.metric(label="Model Confidence", value=f"{result['conf']*100:.2f}%")

    # === Visual Outputs ===
    st.subheader("🎯 Grad-CAM Visual Analysis")
    cols = st.columns(3)
    cols[0].image(result["original"], caption="Original", use_container_width=True)
    cols[1].image(result["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)
    cols[2].image(result["overlay"], caption="Overlay", use_container_width=True)

    # === AI Explanation ===
    st.subheader("🧠 AI Interpretation")
    st.info(result["observation"])

    st.subheader("📌 Clinical Recommendation")
    st.warning(result["note"])

    # === Technical Summary ===
    st.subheader("🧪 Technical Summary")
    st.code(f"""
Model        : ResNet18
Prediction   : {'Normal' if result['pred'] == 0 else 'Abnormal'}
Confidence   : {result['conf']*100:.2f}%
Focus Area   : {result['focus_percent']}%
Attention XY : {result['center']}
Generated    : Live Inference
""", language="yaml")

    # === Download Button for Dynamic Report ===
    st.download_button(
        label="📄 Download Full PDF Report",
        data=result["pdf_buffer"],
        file_name="Patch_GradCAM_Report.pdf",
        mime="application/pdf"
    )

