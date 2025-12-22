import pandas as pd

df = pd.read_csv("all_haiku.csv")

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'hash'], inplace=True)

df = df.rename(columns={"0": "line1", "1": "line2", "2": "line3"})

df.to_csv('haiku_dataset_hjhalani30.csv', index=False)