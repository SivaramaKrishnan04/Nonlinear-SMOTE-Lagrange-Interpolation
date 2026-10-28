
import pandas as pd

# 1. Load dataset
file_path = "Suicidal Tweet Detection Dataset.csv"

try:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="latin1")

print("Original dataset shape:", df.shape)

# 2. Remove extra spaces from column names
df.columns = df.columns.str.strip()

# 3. Check required columns
if "Tweet" not in df.columns or "Suicide" not in df.columns:
    raise ValueError(
        "Expected columns 'Tweet' and 'Suicide'. "
        f"Found: {df.columns.tolist()}"
    )

# 4. Clean target labels
df["Suicide"] = df["Suicide"].astype("string").str.strip()

# 5. Remove rows with missing target labels
df = df.dropna(subset=["Suicide"]).copy()

# 6. Remove missing and empty tweets
df["Tweet"] = df["Tweet"].fillna("").astype(str).str.strip()
df = df[df["Tweet"] != ""].copy()

# 7. Remove exact duplicate tweet-label pairs
before = len(df)
df = df.drop_duplicates(subset=["Tweet", "Suicide"]).copy()
print("Duplicate tweet-label pairs removed:", before - len(df))

# 8. Display cleaned dataset information
print("\nCleaned dataset shape:", df.shape)

print("\nMissing values:")
print(df[["Tweet", "Suicide"]].isnull().sum())

print("\nClass distribution:")
print(df["Suicide"].value_counts())

print("\nClass percentages:")
print((df["Suicide"].value_counts(normalize=True) * 100).round(2))

# 9. Save cleaned dataset
output_file = "cleaned_tweets.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nCleaned dataset saved as:", output_file)
print("\nFirst five cleaned rows:")
print(df.head())
