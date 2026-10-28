
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
    classification_report,
    confusion_matrix
)

# 1. Load cleaned dataset
df = pd.read_csv("cleaned_tweets.csv", encoding="utf-8-sig")

# 2. Prepare input and target
X = df["Tweet"].fillna("").astype(str)
y = df["Suicide"].astype(str).str.strip()

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())

# 3. Split before fitting TF-IDF
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=3000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=2000)

start = time.perf_counter()
model.fit(X_train_tfidf, y_train)
training_time = time.perf_counter() - start

# 6. Predict test data
start = time.perf_counter()
y_pred = model.predict(X_test_tfidf)
prediction_time = time.perf_counter() - start

# 7. Evaluate
minority_label = "Potential Suicide post"

print("\n===== BASELINE RESULTS =====")
print("Accuracy:",
      round(accuracy_score(y_test, y_pred), 4))

print("Precision:",
      round(precision_score(
          y_test, y_pred,
          pos_label=minority_label,
          zero_division=0
      ), 4))

print("Recall:",
      round(recall_score(
          y_test, y_pred,
          pos_label=minority_label,
          zero_division=0
      ), 4))

print("F1-score:",
      round(f1_score(
          y_test, y_pred,
          pos_label=minority_label,
          zero_division=0
      ), 4))

print("Balanced accuracy:",
      round(balanced_accuracy_score(y_test, y_pred), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Potential Suicide post", "Not Suicide post"]
))

print("\nTraining time:", round(training_time, 4), "seconds")
print("Prediction time:", round(prediction_time, 4), "seconds")

# 8. Save baseline metrics
results = pd.DataFrame([{
    "Method": "Baseline - No Oversampling",
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
}])

results.to_csv("baseline_results.csv", index=False)
print("\nSaved results to baseline_results.csv")
