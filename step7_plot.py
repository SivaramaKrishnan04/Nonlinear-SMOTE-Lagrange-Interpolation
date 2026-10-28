
import pandas as pd
import matplotlib.pyplot as plt

# Load comparison results
df = pd.read_csv("step6_comparison_results.csv")

metrics = [
    "Accuracy",
    "Recall",
    "F1",
    "Balanced Accuracy"
]

# Plot each metric
ax = df.set_index("Method")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0.80, 1.00),
    width=0.8
)

ax.set_title("Performance Comparison of Oversampling Methods")
ax.set_ylabel("Score")
ax.set_xlabel("Method")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(title="Evaluation Metric")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("step7_performance_comparison.png", dpi=300)
plt.show()

print("Chart saved to step7_performance_comparison.png")
