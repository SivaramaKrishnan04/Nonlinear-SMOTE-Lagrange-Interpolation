
import pandas as pd

file_path = "Suicidal Tweet Detection Dataset.csv"

try:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="latin1")

print("===== DATASET INFORMATION =====")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nClass distribution:")
print(df["Suicide"].astype("string").str.strip().value_counts())

print("\nFirst five rows:")
print(df.head())
