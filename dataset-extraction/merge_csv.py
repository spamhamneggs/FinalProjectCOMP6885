import pandas as pd
import re

csv_files = ["datasets_csv/haiku_dataset_huanggab.csv", "datasets_csv/haiku_dataset_statworx.csv", "datasets_csv/haiku_dataset_hjhalani30.csv", "datasets_csv/haiku_dataset_bfbarry.csv"]

# Read and concatenate all CSVs
df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# Normalize Data
df = df.map(lambda s: s.lower() if isinstance(s, str) else s)
df = df.map(lambda s: re.sub(r'[^0-9a-zA-Z\s]+', '', s) if isinstance(s, str) else s)
df = df.map(lambda s: re.sub(r'\s+', ' ', s) if isinstance(s, str) else s)
df = df.map(lambda s: s.strip() if isinstance(s, str) else s)

# Remove duplicates and NaNs
df = df.dropna()
df = df.drop_duplicates(subset=['line1', 'line2', 'line3'])

# Sort by source for easier inspection
df = df.sort_values(by=['source'])

# Save to a single CSV
df.to_csv("haiku_dataset_merged.csv", index=False)