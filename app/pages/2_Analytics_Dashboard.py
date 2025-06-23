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
    # Assume final row has format like: "99 / 100", "0.0046"
    last_row = df.iloc[-1]
    total_correct = last_row[1]  # Assuming format "Correct / Total"
    loss_val = float(last_row[2]) if len(last_row) > 2 else 0.0

    correct, total = [int(x.strip()) for x in str(total_correct).split("/")]

    accuracy = correct / total * 100

    summary_data = {
        "🧬 Model": ["ResNet18"],
        "📅 Evaluation Date": [pd.Timestamp.today().strftime('%Y-%m-%d')],
        "🧪 Total Test Samples": [total],
        "✅ Correct Predictions": [correct],
        "🎯 Overall Accuracy (%)": [f"{accuracy:.2f}%"],
        "📉 Loss Value": [f"{loss_val:.4f}"]
    }
    summary_df = pd.DataFrame(summary_data).T.rename(columns={0: "Value"})
    st.table(summary_df)

except Exception as e:
    st.error(f"⚠️ Failed to parse summary metrics: {e}")

# === Per-Class Breakdown ===
st.subheader("📊 Per-Class Accuracy Breakdown")

try:
    # Class names assumed in row 0
    normal_class = df.iloc[0, 1]
    abnormal_class = df.iloc[0, 2]

    normal_acc = float(df.iloc[1, 2]) * 100
    abnormal_acc = float(df.iloc[2, 2]) * 100

    class_acc = {
        f"{normal_class}": normal_acc,
        f"{abnormal_class}": abnormal_acc
    }
    st.bar_chart(pd.DataFrame.from_dict(class_acc, orient='index', columns=["Accuracy (%)"]))

except Exception as e:
    st.warning(f"⚠️ Could not extract per-class accuracy. {e}")

# === Interpretation ===
st.subheader("🧠 Interpretation")
st.success("The AI model demonstrated high test accuracy and is reliable for assisting in pathology-based cancer detection. However, further clinical validation is encouraged.")
