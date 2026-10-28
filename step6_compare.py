
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score
)
from imblearn.over_sampling import SMOTE

# Import your existing Lagrange implementation
from step5_lagrange import lagrange_oversample

RANDOM_STATE = 42

# 1. Load cleaned dataset
df = pd.read_csv("cleaned_tweets.csv")
df = df.dropna(subset=["Tweet", "Suicide"])
df["Tweet"] = df["Tweet"].astype(str)

X_text = df["Tweet"]
y = df["Suicide"]

# 2. Split data before any oversampling
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# 3. TF-IDF: fit on training data only
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=3000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("\nOriginal training class counts:")
print(y_train.value_counts())

# 4. Compare methods
experiments = [
    ("Baseline", None),
    ("SMOTE", "smote"),
    ("Lagrange degree 2", 2),
    ("Lagrange degree 3", 3)
]

results = []

for method_name, method in experiments:
    print(f"\nRunning: {method_name}")

    if method is None:
        X_resampled, y_resampled = X_train, y_train

    elif method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_resampled, y_resampled = sampler.fit_resample(
            X_train, y_train
        )

    else:
        X_resampled, y_resampled = lagrange_oversample(
            X_train,
            y_train,
            degree=method,
            neighbors=5,
            random_state=RANDOM_STATE
        )

    model = LogisticRegression(max_iter=1000)

    start = time.perf_counter()
    model.fit(X_resampled, y_resampled)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - start

    # Use Potential Suicide post as the positive class
    positive_label = "Potential Suicide post"
    y_test_binary = (y_test == positive_label).astype(int)
    y_pred_binary = (y_pred == positive_label).astype(int)

    y_score = model.predict_proba(X_test)[
        :, list(model.classes_).index(positive_label)
    ]

    row = {
        "Method": method_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test_binary, y_pred_binary, zero_division=0
        ),
        "Recall": recall_score(
            y_test_binary, y_pred_binary, zero_division=0
        ),
        "F1": f1_score(
            y_test_binary, y_pred_binary, zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Average Precision": average_precision_score(
            y_test_binary, y_score
        ),
        "Training Time (seconds)": training_time,
        "Prediction Time (seconds)": prediction_time
    }

    results.append(row)
    print(pd.Series(row).to_string())

# 5. Save and display comparison
results_df = pd.DataFrame(results)
results_df.to_csv("step6_comparison_results.csv", index=False)

print("\n===== FINAL COMPARISON =====")
print(results_df.round(4).to_string(index=False))
print("\nSaved to step6_comparison_results.csv")
