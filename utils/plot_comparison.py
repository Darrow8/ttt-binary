"""Plot answer distribution histograms: base model vs GRPO checkpoint."""

import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("runs/base_model_inference/20260310_015526/results.json") as f:
    base_data = json.load(f)

with open("runs/local_inference/20260310_015959/results.json") as f:
    grpo_data = json.load(f)

# Extract answers
base_answers = [r.get("answer", "") for r in base_data["results"] if r.get("answer")]
grpo_answers = [r.get("answer", "") for r in grpo_data["results"] if r.get("answer")]

base_counts = Counter(base_answers)
grpo_counts = Counter(grpo_answers)

# Get top answers across both (union of top 15 from each)
top_answers = set()
for ans, _ in base_counts.most_common(15):
    top_answers.add(ans)
for ans, _ in grpo_counts.most_common(15):
    top_answers.add(ans)

# Sort: put "214" (correct) and "500" (majority) first, then by total count
def sort_key(ans):
    if ans == "214":
        return (0, 0)
    if ans == "500":
        return (0, 1)
    return (1, -(base_counts.get(ans, 0) + grpo_counts.get(ans, 0)))

answers_sorted = sorted(top_answers, key=sort_key)

# Build bar data
base_vals = [base_counts.get(a, 0) for a in answers_sorted]
grpo_vals = [grpo_counts.get(a, 0) for a in answers_sorted]

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(answers_sorted))
width = 0.35

bars1 = ax.bar(x - width/2, base_vals, width, label=f"Base Model (n={len(base_answers)})", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width/2, grpo_vals, width, label=f"GRPO Checkpoint (n={len(grpo_answers)})", color="#DD8452", alpha=0.85)

# Highlight correct answer
for i, ans in enumerate(answers_sorted):
    if ans == "214":
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="green")
        ax.annotate("correct", (i, max(base_counts.get(ans, 0), grpo_counts.get(ans, 0)) + 5),
                    ha="center", fontsize=9, color="green", fontweight="bold")

# Add count labels on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, str(int(h)),
                ha="center", va="bottom", fontsize=7)
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, str(int(h)),
                ha="center", va="bottom", fontsize=7)

ax.set_xlabel("Answer")
ax.set_ylabel("Count (out of 500)")
ax.set_title("Answer Distribution: Base Model vs GRPO Checkpoint (500 samples each)")
ax.set_xticks(x)
ax.set_xticklabels(answers_sorted, rotation=45, ha="right", fontsize=9)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("runs/answer_comparison_500.png", dpi=150)
print("Saved to runs/answer_comparison_500.png")
