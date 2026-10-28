
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix

from step5_lagrange import lagrange_oversample


RANDOM_STATE = 42
MINORITY_LABEL = "Potential Suicide post"


# 1. Load the cleaned dataset
df = pd.read_csv("cleaned_tweets.csv")
df = df.dropna(subset=["Tweet", "Suicide"])
df["Tweet"] = df["Tweet"].astype(str)

X_text = df["Tweet"]
y = df["Suicide"]


# 2. Use the same train/test split as earlier experiments
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

# Original minority-class feature vectors
X_minority = X_train[
    y_train.to_numpy() == MINORITY_LABEL
]

original_count = X_train.shape[0]

print("\nOriginal training shape:", X_train.shape)
print("Original minority samples:", X_minority.shape[0])


# 4. Generate synthetic samples using Lagrange interpolation
X_lagrange, y_lagrange = lagrange_oversample(
    X_train,
    y_train,
    degree=2,
    neighbors=5,
    random_state=RANDOM_STATE
)

X_lagrange = csr_matrix(X_lagrange)
X_lagrange_synthetic = X_lagrange[original_count:]


# 5. Generate synthetic samples using standard SMOTE
smote = SMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_smote, y_smote = smote.fit_resample(X_train, y_train)

X_smote = csr_matrix(X_smote)
X_smote_synthetic = X_smote[original_count:]


# 6. Define a common quality-check function
def analyse_samples(method_name, X_synthetic, X_original):

    X_synthetic = csr_matrix(X_synthetic)

    # Check numeric validity
    finite = bool(np.isfinite(X_synthetic.data).all())
    nonnegative = bool((X_synthetic.data >= 0).all())

    # Check vector norms
    norms = np.sqrt(
        np.asarray(
            X_synthetic.multiply(X_synthetic).sum(axis=1)
        ).ravel()
    )

    # Compare each synthetic sample with its nearest original
    # minority-class sample using cosine distance
    X_original_norm = normalize(X_original)
    X_synthetic_norm = normalize(X_synthetic)

    nearest = NearestNeighbors(
        n_neighbors=1,
        metric="cosine"
    )

    nearest.fit(X_original_norm)

    distances, _ = nearest.kneighbors(X_synthetic_norm)
    similarities = 1 - distances.ravel()

    # Count samples extremely similar to original samples
    almost_identical = int(
        np.sum(similarities >= 0.999999)
    )

    # Check duplicate synthetic vectors after rounding
    rounded = pd.DataFrame(
        X_synthetic_norm.toarray()
    ).round(10)

    duplicate_count = int(rounded.duplicated().sum())

    result = {
        "Method": method_name,
        "Synthetic samples": X_synthetic.shape[0],
        "All values finite": finite,
        "All values nonnegative": nonnegative,
        "Minimum vector norm": norms.min() if len(norms) else np.nan,
        "Maximum vector norm": norms.max() if len(norms) else np.nan,
        "Mean nearest-original cosine similarity":
            similarities.mean() if len(similarities) else np.nan,
        "Minimum cosine similarity":
            similarities.min() if len(similarities) else np.nan,
        "Maximum cosine similarity":
            similarities.max() if len(similarities) else np.nan,
        "Almost identical to original": almost_identical,
        "Duplicate synthetic rows": duplicate_count
    }

    return result


# 7. Analyse both methods
results = []

results.append(
    analyse_samples(
        "Lagrange degree 2",
        X_lagrange_synthetic,
        X_minority
    )
)

results.append(
    analyse_samples(
        "Standard SMOTE",
        X_smote_synthetic,
        X_minority
    )
)


# 8. Display and save the comparison
results_df = pd.DataFrame(results)

print("\n===== STEP 11: SYNTHETIC SAMPLE VALIDATION =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    "step11_synthetic_validation_results.csv",
    index=False
)

print("\nSaved: step11_synthetic_validation_results.csv")
