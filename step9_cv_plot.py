
import pandas as pd
import matplotlib.pyplot as plt

# Load fold-level cross-validation results
df = pd.read_csv("step8_cv_fold_results.csv")

# Calculate mean and standard deviation of F1-score
summary = df.groupby("Method")["F1"].agg(["mean", "std"])

print("\nF1-score across five folds:")
print(summary.round(4))

# Plot mean F1-score with standard deviation
plt.figure(figsize=(10, 6))

plt.bar(
    summary.index,
    summary["mean"],
    yerr=summary["std"],
    capsize=6
)

plt.title("5-Fold Cross-Validation: F1-Score Comparison")
plt.xlabel("Method")
plt.ylabel("Mean F1-Score")
plt.ylim(0.85, 0.95)
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("step9_cv_f1_comparison.png", dpi=300)
plt.show()

print("\nChart saved to step9_cv_f1_comparison.png")
