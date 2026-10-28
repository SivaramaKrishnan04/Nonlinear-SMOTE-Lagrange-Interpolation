
import time
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
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

# 1. Load cleaned dataset
df = pd.read_csv("cleaned_tweets.csv", encoding="utf-8-sig")

X = df["Tweet"].fillna("").astype(str)
y = df["Suicide"].astype(str).str.strip()

minority_label = "Potential Suicide post"

# 2. Use exactly the same split as the baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows before SMOTE:", len(X_train))
print("\nTraining classes before SMOTE:")
print(y_train.value_counts())

# 3. Fit TF-IDF on training data only
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=3000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Apply SMOTE to training data only
smote = SMOTE(random_state=42, k_neighbors=5)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_tfidf, y_train
)

print("\nTraining rows after SMOTE:", X_train_resampled.shape[0])
print("\nTraining classes after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# 5. Train the same classifier
model = LogisticRegression(max_iter=2000)

start = time.perf_counter()
model.fit(X_train_resampled, y_train_resampled)
training_time = time.perf_counter() - start

# 6. Predict the untouched test set
start = time.perf_counter()
y_pred = model.predict(X_test_tfidf)
prediction_time = time.perf_counter() - start

# 7. Evaluate results
metrics = {
    "Method": "Standard SMOTE",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, pos_label=minority_label, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, pos_label=minority_label, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, y_pred, pos_label=minority_label, zero_division=0
    ),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    "Training Time": training_time,
    "Prediction Time": prediction_time
}

print("\n===== SMOTE RESULTS =====")
for name, value in metrics.items():
    if isinstance(value, (float, int)):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Potential Suicide post", "Not Suicide post"]
))

# 8. Save SMOTE metrics
pd.DataFrame([metrics]).to_csv("smote_results.csv", index=False)
print("\nSaved results to smote_results.csv")
