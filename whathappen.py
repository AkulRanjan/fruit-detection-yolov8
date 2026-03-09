import pandas as pd
df = pd.read_csv('runs/segment/fruit_seg/results.csv')
df.columns = df.columns.str.strip()
print(df.to_string())