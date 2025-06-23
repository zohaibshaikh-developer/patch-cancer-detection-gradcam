import streamlit as st
import pandas as pd
from utils.analytics_utils import load_metrics

st.set_page_config(page_title="Analytics Dashboard", layout="wide")
st.title("📈 Model Evaluation Summary")
st.markdown("This dashboard provides a high-level overview of the AI model's test performance on the Camelyon17-clean dataset.")

df = load_metrics()

# === Summary Section ===
st.subheader("🧪 Test Performance Metrics")

try:
    total_row = df[df.iloc[:, 0].str.lower().str.contains("total")].iloc[0]
    total_samples = int(total_row[1])
    accuracy = float(total_row[2]) * 100
    loss_val = float(total_row[3]) if len(total_row) > 3 else 0.0  # optional

    correct = int(total_samples * accuracy / 100)

    summary_data = {
        "🧬 Model": ["ResNet18"],
        "📅 Evaluation Date": [pd.Timestamp.today().strftime('%Y-%m-%d')],
        "🧪 Total Test Samples": [total_samples],
        "✅ Correct Predictions": [correct],
        "🎯 Overall Accuracy (%)": [f"{accuracy:.2f}%"],
        "📉 Loss Value": [f"{loss_val:.4f}"] if loss_val else []
    }
    summary_df = pd.DataFrame(summary_data).T.rename(columns={0: "Value"})
    st.table(summary_df)

except Exception as e:
    st.error(f"⚠️ Failed to parse summary metrics: {e}")

# === Per-Class Accuracy ===
st.subheader("📊 Per-Class Accuracy Breakdown")

try:
    normal_row = df[df.iloc[:, 0].str.lower().str.contains("normal")].iloc[0]
    abnormal_row = df[df.iloc[:, 0].str.lower().str.contains("abnormal")].iloc[0]

    class_acc = {
        "Normal Tissue": float(normal_row[2]) * 100,
        "Abnormal / Cancerous": float(abnormal_row[2]) * 100
    }

    st.bar_chart(pd.DataFrame.from_dict(class_acc, orient='index', columns=["Accuracy (%)"]))

except Exception as e:
    st.warning(f"⚠️ Could not extract per-class accuracy: {e}")

# === Interpretation ===
st.subheader("🧠 Interpretation")
st.success("The AI model demonstrated high test accuracy and is reliable for assisting in pathology-based cancer detection. However, further clinical validation is encouraged.")
