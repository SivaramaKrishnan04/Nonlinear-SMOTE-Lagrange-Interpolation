
import time
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
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

# Import only the function definition, without running step5's experiment.
# We will instead use a separate helper module in the next step if needed.
from step5_lagrange import lagrange_oversample

RANDOM_STATE = 42
POSITIVE_LABEL = "Potential Suicide post"

# 1. Load the cleaned dataset
df = pd.read_csv("cleaned_tweets.csv")
df = df.dropna(subset=["Tweet", "Suicide"])
df["Tweet"] = df["Tweet"].astype(str)

X = df["Tweet"].reset_index(drop=True)
y = df["Suicide"].reset_index(drop=True)

# 2. Define 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

methods = [
    ("Baseline", None),
    ("SMOTE", "smote"),
    ("Lagrange degree 2", 2),
    ("Lagrange degree 3", 3)
]

all_results = []

# 3. Run each method through the same folds
for method_name, method in methods:
    print(f"\nRunning cross-validation: {method_name}")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train_text = X.iloc[train_idx]
        X_test_text = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fit TF-IDF on training fold only
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=3000
        )

        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        # Oversample training data only
        if method is None:
            X_resampled, y_resampled = X_train, y_train

        elif method == "smote":
            sampler = SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=5
            )
            X_resampled, y_resampled = sampler.fit_resample(
                X_train, y_train
            )

        else:
            X_resampled, y_resampled = lagrange_oversample(
                X_train,
                y_train,
                degree=method,
                neighbors=5,
                random_state=RANDOM_STATE + fold
            )

        # Train classifier
        model = LogisticRegression(max_iter=1000)

        start = time.perf_counter()
        model.fit(X_resampled, y_resampled)
        train_time = time.perf_counter() - start

        # Evaluate on the untouched fold
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - start

        y_test_binary = (y_test == POSITIVE_LABEL).astype(int)
        y_pred_binary = (y_pred == POSITIVE_LABEL).astype(int)

        positive_index = list(model.classes_).index(POSITIVE_LABEL)
        y_score = model.predict_proba(X_test)[:, positive_index]

        all_results.append({
            "Method": method_name,
            "Fold": fold,
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
            "Training Time": train_time,
            "Prediction Time": predict_time
        })

        print(
            f"Fold {fold}: "
            f"F1={all_results[-1]['F1']:.4f}, "
            f"Recall={all_results[-1]['Recall']:.4f}"
        )

# 4. Save fold-level results
fold_df = pd.DataFrame(all_results)
fold_df.to_csv("step8_cv_fold_results.csv", index=False)

# 5. Calculate mean and standard deviation
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
    "Average Precision"
]

summary = fold_df.groupby("Method")[metric_columns].agg(["mean", "std"])
summary.to_csv("step8_cv_summary.csv")

print("\n===== 5-FOLD CROSS-VALIDATION SUMMARY =====")
print(summary.round(4).to_string())
print("\nSaved fold results to step8_cv_fold_results.csv")
print("Saved summary to step8_cv_summary.csv")
