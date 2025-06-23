import streamlit as st
from utils.model_loader import predict_and_visualize
from PIL import Image, UnidentifiedImageError

# === Streamlit Page Config ===
st.set_page_config(page_title="Upload and Predict", layout="wide")
st.title("📤 Upload Histopathology Patch")

st.markdown("""
Upload a patch from the **Camelyon17 dataset** to perform **AI-based cancer detection** using a ResNet18 model with Grad-CAM interpretability.
""")

# === File Uploader ===
uploaded = st.file_uploader("🖼 Upload a .png or .jpg patch image", type=["png", "jpg", "jpeg"])

if uploaded:
    try:
        # === Convert to PIL and Display Uploaded Patch ===
        image_pil = Image.open(uploaded).convert("RGB")
        st.image(image_pil, caption="🧩 Uploaded Patch", width=250)

        # === Run Inference + Grad-CAM ===
        with st.spinner("🔍 Running model inference and Grad-CAM..."):
            result = predict_and_visualize(image_pil)

        # === Prediction Output ===
        st.subheader("🧠 Prediction Results")
        label = "✅ Normal Tissue" if result["pred"] == 0 else "❗ Abnormal / Cancerous"
        color = "green" if result["pred"] == 0 else "red"
        st.markdown(
            f"### Prediction: <span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True
        )
        st.metric(label="Model Confidence", value=f"{result['conf']*100:.2f}%")

        # === Grad-CAM Visuals ===
        st.subheader("🎯 Grad-CAM Visual Analysis")
        cols = st.columns(3)
        cols[0].image(result["original"], caption="Original Image", use_container_width=True)
        cols[1].image(result["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)
        cols[2].image(result["overlay"], caption="Overlay Visualization", use_container_width=True)

        # === Interpretability Output ===
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
            Focus Area   : {result['focus_percent']}
            Attention XY : {result['center']}
            Generated    : Live Inference
            """, language="yaml")

        # === PDF Report Download ===
        st.download_button(
            label="📄 Download Full PDF Report",
            data=result["pdf_buffer"],
            file_name="Patch_GradCAM_Report.pdf",
            mime="application/pdf"
        )

    except UnidentifiedImageError:
        st.error("❌ Uploaded file is not a valid image. Please upload a .png or .jpg image.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {str(e)}")

else:
    st.info("Please upload a valid image file to begin analysis.")
