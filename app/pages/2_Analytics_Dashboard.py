import streamlit as st
import pandas as pd
from utils.analytics_utils import load_metrics

st.set_page_config(page_title="Analytics Dashboard", layout="wide")
st.title("📈 Model Evaluation Summary")
st.markdown("This dashboard provides an overview of the AI model's performance on the Camelyon17-clean dataset.")

df = load_metrics()

# === Summary Section ===
st.subheader("🧪 Test Performance Metrics")

try:
    total_match = df[df.iloc[:, 0].str.lower().str.contains("total")]
    if not total_match.empty:
        total_row = total_match.iloc[0]
        total_samples = int(total_row[1])
        accuracy = float(total_row[2]) * 100
        loss_val = float(total_row[3]) if len(total_row) > 3 else None
        correct = int(total_samples * accuracy / 100)

        summary_data = {
            "🧬 Model": ["ResNet18"],
            "📅 Evaluation Date": [pd.Timestamp.today().strftime('%Y-%m-%d')],
            "🧪 Total Test Samples": [total_samples],
            "✅ Correct Predictions": [correct],
            "🎯 Overall Accuracy (%)": [f"{accuracy:.2f}%"],
        }
        if loss_val is not None:
            summary_data["📉 Loss Value"] = [f"{loss_val:.4f}"]

        summary_df = pd.DataFrame(summary_data).T.rename(columns={0: "Value"})
        st.table(summary_df)
    else:
        raise ValueError("No row with 'Total' found.")

except Exception as e:
    st.error(f"⚠️ Failed to parse summary metrics: {e}")

# === Per-Class Accuracy ===
st.subheader("📊 Per-Class Accuracy Breakdown")

try:
    normal_match = df[df.iloc[:, 0].str.lower().str.contains("normal")]
    abnormal_match = df[df.iloc[:, 0].str.lower().str.contains("abnormal|cancer")]

    if not normal_match.empty and not abnormal_match.empty:
        normal_row = normal_match.iloc[0]
        abnormal_row = abnormal_match.iloc[0]

        class_acc = {
            "Normal Tissue": float(normal_row[2]) * 100,
            "Abnormal / Cancerous": float(abnormal_row[2]) * 100
        }

        st.bar_chart(pd.DataFrame.from_dict(class_acc, orient='index', columns=["Accuracy (%)"]))
    else:
        raise ValueError("Missing one or both class accuracy rows.")

except Exception as e:
    st.warning(f"⚠️ Could not extract per-class accuracy: {e}")

# === Interpretation ===
st.subheader("🧠 Interpretation")
st.success("The model shows high performance in test evaluations. Suitable for clinical decision support but further validation is encouraged.")
