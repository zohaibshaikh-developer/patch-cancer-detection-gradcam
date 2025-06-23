import streamlit as st
import pandas as pd
from app.utils.analytics_utils import load_metrics

st.title("📊 Analytics Dashboard")

df = load_metrics()

st.subheader("🧪 Summary Test Metrics")
st.write(df.iloc[0])  # Shows total samples, accuracy, loss

# Optionally plot a pie chart of per-class accuracy
st.subheader("🎯 Per-Class Accuracy")
class_acc = {
    df.columns[1]: float(df.iloc[1, 2]),
    df.columns[2]: float(df.iloc[2, 2])
}
st.bar_chart(class_acc)
