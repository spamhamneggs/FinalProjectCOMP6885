import pandas as pd

# Read the text file
with open("lines.txt", "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

# Split each haiku line into its three parts
data = []
for line in raw_lines:
    line = line.strip().rstrip("$")  # remove trailing spaces and the "$"
    parts = [part.strip() for part in line.split("/")]

    if len(parts) == 3:  # only take proper 3-part haikus
        data.append(parts)

# Create DataFrame
df = pd.DataFrame(data, columns=["line1", "line2", "line3"])

# Add source column
df['source'] = 'bfbarry'

# Save to CSV
df.to_csv("haiku_dataset_bfbarry.csv", index=False, encoding="utf-8")
