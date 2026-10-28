
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from scipy import sparse

# ==================================================
# 1. CONFIGURATION
# ==================================================

RANDOM_STATE = 42
DEGREE = 2       # Try 2, then 3
NEIGHBORS = 5    # Number of minority nearest neighbors

# ==================================================
# 2. CUSTOM LAGRANGE OVERSAMPLER
# ==================================================

def lagrange_oversample(X, y, degree=2, neighbors=5, random_state=42):
    """
    Generate minority-class feature vectors using Lagrange interpolation.

    X: numerical feature matrix (TF-IDF vectors)
    y: class labels
    degree: polynomial degree, at least 2
    neighbors: number of minority neighbors to consider
    """

    if degree < 2:
        raise ValueError("Degree must be at least 2.")

    if neighbors < degree:
        raise ValueError("Neighbors must be >= degree.")

    rng = np.random.default_rng(random_state)
    y_array = np.asarray(y)
    classes, counts = np.unique(y_array, return_counts=True)

    if len(classes) != 2:
        raise ValueError("This implementation requires exactly 2 classes.")

    minority_class = classes[np.argmin(counts)]
    minority_count = int(counts.min())
    majority_count = int(counts.max())

    needed = majority_count - minority_count

    if needed == 0:
        return X, y_array

    minority_indices = np.flatnonzero(y_array == minority_class)
    X_min = X[minority_indices]

    if len(minority_indices) <= degree:
        raise ValueError(
            f"Need at least {degree + 1} minority samples."
        )

    # Find nearest minority-class neighbors.
    k = min(neighbors + 1, len(minority_indices))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_min)
    neighbor_indices = nn.kneighbors(
        X_min, return_distance=False
    )

    nodes = np.arange(degree + 1, dtype=float)
    generated = []
    generated_checks = []

    attempts = 0
    max_attempts = max(1000, needed * 100)

    while len(generated) < needed and attempts < max_attempts:
        attempts += 1

        # Select an anchor and degree of its nearby minority samples.
        anchor_index = int(rng.integers(len(minority_indices)))
        candidates = neighbor_indices[anchor_index, 1:]

        if len(candidates) < degree:
            continue

        selected_neighbors = rng.choice(
            candidates, size=degree, replace=False
        )
        selected_indices = np.concatenate(
            ([anchor_index], selected_neighbors)
        )

        selected = X_min[selected_indices].toarray()
        anchor = selected[0]

        # Random polynomial evaluation point in [0, 1].
        t = float(rng.uniform(0.0, 1.0))

        # Calculate Lagrange basis weights.
        weights = np.ones(degree + 1)

        for j in range(degree + 1):
            for m in range(degree + 1):
                if j != m:
                    weights[j] *= (
                        (t - nodes[m]) / (nodes[j] - nodes[m])
                    )

        # Evaluate the vector-valued polynomial.
        synthetic = weights @ selected

        # TF-IDF features should be nonnegative.
        synthetic = np.maximum(synthetic, 0.0)

        # Normalize vector length.
        norm = np.linalg.norm(synthetic)

        if not np.isfinite(norm) or norm < 1e-12:
            continue

        synthetic = synthetic / norm

        # Reject samples too far from the selected local neighborhood.
        local_radius = max(
            np.linalg.norm(row - anchor)
            for row in selected[1:]
        )
        anchor_distance = np.linalg.norm(synthetic - anchor)

        if local_radius > 0 and anchor_distance > 1.5 * local_radius:
            continue

        # Reject near-duplicates among recent generated samples.
        if any(
            np.linalg.norm(synthetic - previous) < 1e-6
            for previous in generated_checks[-200:]
        ):
            continue

        generated.append(synthetic)
        generated_checks.append(synthetic)

    if len(generated) < needed:
        warnings.warn(
            f"Generated only {len(generated)} of {needed} requested "
            "samples. Class balance may remain imperfect."
        )

    if not generated:
        return X, y_array

    generated_array = np.vstack(generated)

    if sparse.issparse(X):
        X_result = sparse.vstack(
            [X, sparse.csr_matrix(generated_array)],
            format="csr"
        )
    else:
        X_result = np.vstack([X, generated_array])

    y_result = np.concatenate([
        y_array,
        np.full(len(generated), minority_class, dtype=y_array.dtype)
    ])

    return X_result, y_result


# ==================================================
# 3. LOAD DATA
# ==================================================

df = pd.read_csv("cleaned_tweets.csv", encoding="utf-8-sig")

X = df["Tweet"].fillna("").astype(str)
y = df["Suicide"].astype(str).str.strip()

minority_label = "Potential Suicide post"

# Use the SAME split as baseline and SMOTE.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training rows before interpolation:", len(X_train))
print("\nOriginal training class counts:")
print(y_train.value_counts())

# ==================================================
# 4. TF-IDF
# ==================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=3000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==================================================
# 5. APPLY LAGRANGE OVERSAMPLING TO TRAINING ONLY
# ==================================================

X_resampled, y_resampled = lagrange_oversample(
    X_train_tfidf,
    y_train,
    degree=DEGREE,
    neighbors=NEIGHBORS,
    random_state=RANDOM_STATE
)
print("\n===== SYNTHETIC DATA CHECK =====")
print("Original training samples:", X_train_tfidf.shape[0])
print("Synthetic samples created:",
      X_resampled.shape[0] - X_train_tfidf.shape[0])
print("Original feature shape:", X_train_tfidf.shape)
print("Resampled feature shape:", X_resampled.shape)

values = X_resampled.data if hasattr(X_resampled, "data") else X_resampled.ravel()
print("All feature values are finite:", np.isfinite(values).all())

print("\nTraining rows after interpolation:", X_resampled.shape[0])
print("\nClass counts after interpolation:")
print(pd.Series(y_resampled).value_counts())

# ==================================================
# 6. TRAIN CLASSIFIER
# ==================================================

model = LogisticRegression(max_iter=2000)

start = time.perf_counter()
model.fit(X_resampled, y_resampled)
training_time = time.perf_counter() - start

# ==================================================
# 7. TEST AND EVALUATE
# ==================================================

start = time.perf_counter()
y_pred = model.predict(X_test_tfidf)
prediction_time = time.perf_counter() - start

metrics = {
    "Method": "Lagrange degree " + str(DEGREE),
    "Degree": DEGREE,
    "Neighbors": NEIGHBORS,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred,
        pos_label=minority_label,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred,
        pos_label=minority_label,
        zero_division=0
    ),
    "F1-score": f1_score(
        y_test, y_pred,
        pos_label=minority_label,
        zero_division=0
    ),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    "Average Precision": average_precision_score(
    (y_test == minority_label).astype(int),
    model.predict_proba(X_test_tfidf)[
        :, list(model.classes_).index(minority_label)
    ]
),
    "Training Time": training_time,
    "Prediction Time": prediction_time
}

print("\n===== LAGRANGE RESULTS =====")

for key, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Potential Suicide post", "Not Suicide post"]
))

# ==================================================
# 8. SAVE RESULTS
# ==================================================


output_file = f"lagrange_degree{DEGREE}_results.csv"

pd.DataFrame([metrics]).to_csv(
    output_file,
    index=False
)

print(f"\nSaved results to {output_file}")
