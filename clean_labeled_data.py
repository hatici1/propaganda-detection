import pandas as pd

df = pd.read_csv("labeled_fragments.csv")

# Drop empty text or technique fields
df.dropna(subset=["text", "technique"], inplace=True)
df = df[df["text"].str.strip() != ""]
df = df[df["technique"].str.strip() != ""]

# Drop rows with invalid spans
df = df[df["end"] > df["start"]]

# Optional: drop very short spans
df = df[df["text"].str.len() > 3]

# Drop exact duplicates
df.drop_duplicates(inplace=True)

# Reset index and save
df.reset_index(drop=True, inplace=True)
df.to_csv("labeled_fragments_cleaned.csv", index=False)

print(f"✅ Cleaned dataset saved as labeled_fragments_cleaned.csv ({len(df)} rows)")
