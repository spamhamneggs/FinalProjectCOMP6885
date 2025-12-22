import pandas as pd

df = pd.read_csv("merged_with_keywords.csv")

# Add source column
df['source'] = 'huanggab'

# Split into three columns, filling missing values if needed
lines = df['processed_title'].str.split('/', n=2, expand=True)
lines = lines.reindex(columns=[0, 1, 2])  # Ensure three columns
lines.columns = ['line1', 'line2', 'line3']
df = pd.concat([df, lines], axis=1)

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'id', 'keywords', 'processed_title', 'ups'], inplace=True)

df.to_csv('haiku_dataset_huanggab.csv', index=False)