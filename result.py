import pandas as pd
import os

# Find all results.csv files
base = "runs/segment"
for folder in os.listdir(base):
    csv_path = f"{base}/{folder}/results.csv"
    if os.path.exists(csv_path):
        print(f"\n=== {folder} ===")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        print(f"  Epochs trained : {len(df)}")
        print(f"  Best mAP50     : {df['metrics/mAP50(B)'].max():.4f}")
        print(f"  Best mAP50-95  : {df['metrics/mAP50-95(B)'].max():.4f}")
        print(f"  Final box_loss : {df['train/box_loss'].iloc[-1]:.4f}")
        print(f"  Final cls_loss : {df['train/cls_loss'].iloc[-1]:.4f}")
        print(f"  Final seg_loss : {df['train/seg_loss'].iloc[-1]:.4f}")