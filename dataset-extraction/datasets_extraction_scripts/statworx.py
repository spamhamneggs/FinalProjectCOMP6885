from datasets import load_dataset
import pandas as pd

d = load_dataset('statworx/haiku', split='train')

# Convert to pandas DataFrame
df = d.to_pandas()

# Split the 'text' column into three new columns
df[['line1', 'line2', 'line3']] = df['text'].str.split(' / ', expand=True)

# Drop unnecessary columns
df.drop(columns=['text', 'text_phonemes', 'keywords', 'keyword_phonemes', 'gruen_score', 'text_punc'], inplace=True)

# Save to CSV
df.to_csv('haiku_dataset_statworx.csv', index=False)