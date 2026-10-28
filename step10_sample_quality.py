
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from step5_lagrange import lagrange_oversample

RANDOM_STATE = 42
MINORITY_LABEL = "Potential Suicide post"

# 1. Load cleaned data
df = pd.read_csv("cleaned_tweets.csv")
df = df.dropna(subset=["Tweet", "Suicide"])
df["Tweet"] = df["Tweet"].astype(str)

X_text = df["Tweet"]
y = df["Suicide"]

# 2. Split before oversampling
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# 3. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=3000
)

X_train = vectorizer.fit_transform(X_train_text)

# 4. Generate Lagrange samples
X_resampled, y_resampled = lagrange_oversample(
    X_train,
    y_train,
    degree=2,
    neighbors=5,
    random_state=RANDOM_STATE
)

# The function appends synthetic samples after original samples.
original_count = X_train.shape[0]
synthetic_count = X_resampled.shape[0] - original_count

print("\n===== SAMPLE QUALITY ANALYSIS =====")
print("Original training samples:", original_count)
print("Synthetic samples:", synthetic_count)

if synthetic_count <= 0:
    raise RuntimeError("No synthetic samples were generated.")

# 5. Extract synthetic samples and minority originals
X_synthetic = X_resampled[original_count:]
X_original_minority = X_train[y_train.to_numpy() == MINORITY_LABEL]

# 6. Check finite values and feature ranges
synthetic_values = X_synthetic.data

print(
    "All synthetic values finite:",
    np.isfinite(synthetic_values).all()
)
print(
    "All synthetic values nonnegative:",
    (synthetic_values >= 0).all()
)

row_norms = np.sqrt(
    np.asarray(X_synthetic.multiply(X_synthetic).sum(axis=1)).ravel()
)

print("Minimum synthetic vector norm:", row_norms.min())
print("Maximum synthetic vector norm:", row_norms.max())

# 7. Check whether synthetic vectors duplicate original minority vectors
original_normed = normalize(X_original_minority)
synthetic_normed = normalize(X_synthetic)

nearest = NearestNeighbors(n_neighbors=1, metric="cosine")
nearest.fit(original_normed)

distances, indices = nearest.kneighbors(synthetic_normed)
similarities = 1.0 - distances.ravel()

print("\n===== NEAREST ORIGINAL MINORITY SAMPLE =====")
print("Mean cosine similarity:", round(similarities.mean(), 6))
print("Minimum cosine similarity:", round(similarities.min(), 6))
print("Maximum cosine similarity:", round(similarities.max(), 6))
print(
    "Synthetic samples almost identical to an original "
    "(similarity >= 0.999999):",
    int((similarities >= 0.999999).sum())
)

# 8. Check exact duplicates among synthetic samples
synthetic_df = pd.DataFrame(
    synthetic_normed.toarray()
).round(10)

duplicate_count = int(synthetic_df.duplicated().sum())

print("\n===== SYNTHETIC DUPLICATE CHECK =====")
print("Exact/rounded duplicate synthetic rows:", duplicate_count)
print(
    "Unique synthetic rows:",
    synthetic_count - duplicate_count
)

# 9. Save summary
summary = pd.DataFrame([{
    "Original training samples": original_count,
    "Synthetic samples": synthetic_count,
    "All values finite": bool(np.isfinite(synthetic_values).all()),
    "All values nonnegative": bool((synthetic_values >= 0).all()),
    "Mean nearest-original cosine similarity": similarities.mean(),
    "Minimum nearest-original cosine similarity": similarities.min(),
    "Maximum nearest-original cosine similarity": similarities.max(),
    "Near-duplicate original matches": int(
        (similarities >= 0.999999).sum()
    ),
    "Rounded duplicate synthetic rows": duplicate_count
}])

summary.to_csv("step10_sample_quality_results.csv", index=False)

print("\nSaved: step10_sample_quality_results.csv")
