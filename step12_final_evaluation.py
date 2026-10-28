
import os
import pandas as pd
import matplotlib.pyplot as plt

# Keep all outputs in the current project folder
OUTPUT_DIR = "."

# 1. Final test-set metrics from your experiments
test_results = pd.DataFrame([
    {
        "Method": "Baseline",
        "Accuracy": 0.929775,
        "Precision": 0.941667,
        "Recall": 0.862595,
        "F1-score": 0.900398,
        "Balanced Accuracy": 0.915742,
        "Average Precision": 0.963525
    },
    {
        "Method": "Standard SMOTE",
        "Accuracy": 0.952247,
        "Precision": 0.931818,
        "Recall": 0.938931,
        "F1-score": 0.935361,
        "Balanced Accuracy": 0.949466,
        "Average Precision": 0.962179
    },
    {
        "Method": "Lagrange degree 2",
        "Accuracy": 0.952247,
        "Precision": 0.931818,
        "Recall": 0.938931,
        "F1-score": 0.935361,
        "Balanced Accuracy": 0.949466,
        "Average Precision": 0.962758
    },
    {
        "Method": "Lagrange degree 3",
        "Accuracy": 0.952247,
        "Precision": 0.931818,
        "Recall": 0.938931,
        "F1-score": 0.935361,
        "Balanced Accuracy": 0.949466,
        "Average Precision": 0.963293
    }
])

# 2. Five-fold cross-validation summary
cv_results = pd.DataFrame([
    {
        "Method": "Baseline",
        "Mean Accuracy": 0.9179,
        "Accuracy SD": 0.0162,
        "Mean Precision": 0.9395,
        "Precision SD": 0.0212,
        "Mean Recall": 0.8301,
        "Recall SD": 0.0375,
        "Mean F1-score": 0.8811,
        "F1 SD": 0.0252,
        "Mean Balanced Accuracy": 0.8995,
        "Balanced Accuracy SD": 0.0202,
        "Mean Average Precision": 0.9632,
        "Average Precision SD": 0.0095
    },
    {
        "Method": "Standard SMOTE",
        "Mean Accuracy": 0.9325,
        "Accuracy SD": 0.0056,
        "Mean Precision": 0.9119,
        "Precision SD": 0.0070,
        "Mean Recall": 0.9035,
        "Recall SD": 0.0117,
        "Mean F1-score": 0.9077,
        "F1 SD": 0.0081,
        "Mean Balanced Accuracy": 0.9264,
        "Balanced Accuracy SD": 0.0068,
        "Mean Average Precision": 0.9635,
        "Average Precision SD": 0.0086
    },
    {
        "Method": "Lagrange degree 2",
        "Mean Accuracy": 0.9319,
        "Accuracy SD": 0.0078,
        "Mean Precision": 0.9182,
        "Precision SD": 0.0092,
        "Mean Recall": 0.8944,
        "Recall SD": 0.0135,
        "Mean F1-score": 0.9061,
        "F1 SD": 0.0109,
        "Mean Balanced Accuracy": 0.9240,
        "Balanced Accuracy SD": 0.0089,
        "Mean Average Precision": 0.9634,
        "Average Precision SD": 0.0090
    },
    {
        "Method": "Lagrange degree 3",
        "Mean Accuracy": 0.9302,
        "Accuracy SD": 0.0082,
        "Mean Precision": 0.9152,
        "Precision SD": 0.0072,
        "Mean Recall": 0.8928,
        "Recall SD": 0.0169,
        "Mean F1-score": 0.9038,
        "F1 SD": 0.0119,
        "Mean Balanced Accuracy": 0.9224,
        "Balanced Accuracy SD": 0.0100,
        "Mean Average Precision": 0.9633,
        "Average Precision SD": 0.0093
    }
])

# 3. Save both comparison tables
test_results.to_csv(
    os.path.join(OUTPUT_DIR, "step12_final_test_comparison.csv"),
    index=False
)

cv_results.to_csv(
    os.path.join(OUTPUT_DIR, "step12_final_cv_comparison.csv"),
    index=False
)

print("\n===== FINAL TEST-SET COMPARISON =====")
print(test_results.round(4).to_string(index=False))

print("\n===== FIVE-FOLD CROSS-VALIDATION =====")
print(cv_results.round(4).to_string(index=False))


# 4. Plot test-set metrics
test_plot = test_results.set_index("Method")[
    ["Accuracy", "Precision", "Recall", "F1-score",
     "Balanced Accuracy"]
]

ax = test_plot.plot(kind="bar", figsize=(12, 6))
ax.set_title("Test-Set Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xlabel("Method")
plt.xticks(rotation=15, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("step12_test_metrics_comparison.png", dpi=300)
plt.close()

# 5. Plot cross-validation F1 scores
cv_plot = cv_results.set_index("Method")[
    ["Mean F1-score"]
]

ax = cv_plot.plot(
    kind="bar",
    yerr=cv_results.set_index("Method")["F1 SD"],
    capsize=4,
    figsize=(10, 6),
    legend=False
)

ax.set_title("Five-Fold Cross-Validation: Mean F1-Score")
ax.set_ylabel("Mean F1-score")
ax.set_ylim(0.80, 0.95)
ax.set_xlabel("Method")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("step12_cv_f1_comparison.png", dpi=300)
plt.close()

# 6. Load Step 11 sample quality results, if available
quality_file = "step11_synthetic_validation_results.csv"

if os.path.exists(quality_file):
    quality_results = pd.read_csv(quality_file)
    print("\n===== SYNTHETIC SAMPLE QUALITY =====")
    print(quality_results.to_string(index=False))
else:
    print("\nStep 11 quality CSV was not found.")

print("\nStep 12 completed.")
print("Saved:")
print("- step12_final_test_comparison.csv")
print("- step12_final_cv_comparison.csv")
print("- step12_test_metrics_comparison.png")
print("- step12_cv_f1_comparison.png")
