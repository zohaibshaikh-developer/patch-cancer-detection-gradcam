import pandas as pd
import glob

def load_metrics(path=None):
    metrics_files = sorted(glob.glob("logs/test_metrics_*.csv"))
    if not metrics_files:
        raise FileNotFoundError("No test_metrics CSV found in logs/")
    
    latest = metrics_files[-1]
    df = pd.read_csv(latest)

    # 🔧 Ensure Arrow compatibility
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    return df