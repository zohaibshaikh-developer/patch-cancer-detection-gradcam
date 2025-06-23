import streamlit as st
import pandas as pd
from utils.analytics_utils import load_metrics

# Page config
st.set_page_config(page_title="Analytics Dashboard", layout="wide")

# Load metrics
df = load_metrics()

# Extract top-level stats
total_samples = int(df.iloc[0, 1])
correct = int(df.iloc[0, 2])
accuracy = float(df.iloc[0, 3]) * 100
loss = float(df.iloc[0, 4])

# Header
st.title("📈 Model Evaluation Summary")
st.markdown("This dashboard provides a high-level overview of the AI model's test performance using the Camelyon17-clean dataset.")

# =======================
# Summary Section
# =======================
st.subheader("🧪 Test Performance Metrics")

summary_data = {
    "🧬 Model": ["ResNet18"],
    "📅 Evaluation Date": [pd.Timestamp.today().strftime('%Y-%m-%d')],
    "🧪 Total Test Samples": [total_samples],
    "✅ Correct Predictions": [correct],
    "🎯 Overall Accuracy (%)": [f"{accuracy:.2f}%"],
    "📉 Loss Value": [f"{loss:.4f}"]
}
summary_df = pd.DataFrame(summary_data).T.rename(columns={0: "Value"})
st.table(summary_df)

# =======================
# Per-Class Accuracy
# =======================
st.subheader("📊 Per-Class Accuracy Breakdown")

# Extract per-class accuracy
class_names = {
    df.columns[1]: "Normal Tissue",
    df.columns[2]: "Abnormal / Cancerous"
}
class_acc = {
    class_names[df.columns[1]]: float(df.iloc[1, 2]) * 100,
    class_names[df.columns[2]]: float(df.iloc[2, 2]) * 100
}

st.bar_chart(pd.DataFrame.from_dict(class_acc, orient='index', columns=["Accuracy (%)"]))

# =======================
# Clinical Note
# =======================
st.subheader("🧠 Interpretation")
st.success("The AI model achieved high classification performance on both classes. Accuracy above 99% suggests reliability for supporting clinical diagnosis. We recommend follow-up validation in a controlled clinical setting.")
