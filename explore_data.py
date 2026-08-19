from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create folder for outputs
Path("data").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1: Load the dataset
# ─────────────────────────────────────────────────────────────
# HuggingFace hosts many datasets ready-to-download. This one is
# the Jigsaw Toxic Comment Classification Challenge from Kaggle.
# The load_dataset() function downloads it and caches locally.
# First run: ~2 minutes. Subsequent runs: instant (uses cache).

print("Loading dataset from HuggingFace Hub...")
ds = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")

# A "dataset" from HuggingFace is like a dict with splits:
# usually "train", "validation", "test"
print(f"\nDataset splits available: {list(ds.keys())}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Convert to pandas (easier to explore)
# ─────────────────────────────────────────────────────────────
# HuggingFace datasets are memory-efficient but harder to poke at.
# Pandas is the standard for exploration, so we convert.

train_df = ds["train"].to_pandas()
print(f"\nTrain set has {len(train_df):,} rows")
print(f"Columns: {list(train_df.columns)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Look at actual examples
# ─────────────────────────────────────────────────────────────
# ALWAYS look at real examples before modeling. You'd be shocked
# how many bugs are hiding in "clean" data.

print("\n=== 3 EXAMPLE COMMENTS ===")
for i in range(3):
    row = train_df.iloc[i]
    print(f"\nComment {i+1}:")
    print(f"  Text: {row['comment_text'][:200]}...")
    # Only print the labels that are 1 (toxic in some way)
    labels = [col for col in train_df.columns[2:] if row[col] == 1]
    print(f"  Labels: {labels if labels else ['(clean)']}")

# ─────────────────────────────────────────────────────────────
# STEP 4: Class balance — the crucial question
# ─────────────────────────────────────────────────────────────
# In classification, we need to know: how many of each class?
# If 99% of data is "not toxic," a dumb model that always says
# "not toxic" would be 99% accurate but useless.

label_cols = ["toxic", "severe_toxic", "obscene", "threat",
              "insult", "identity_hate"]

print("\n=== CLASS BALANCE ===")
print("What fraction of comments are labeled each way:\n")
for col in label_cols:
    pct = train_df[col].mean() * 100
    print(f"  {col:15s}: {pct:5.2f}%")

# Add a combined "any_toxic" column: 1 if ANY of the 6 labels is 1
train_df["any_toxic"] = train_df[label_cols].max(axis=1)
overall_toxic_pct = train_df["any_toxic"].mean() * 100
print(f"\n  Any toxic label at all: {overall_toxic_pct:.2f}%")

# ─────────────────────────────────────────────────────────────
# STEP 5: Comment length distribution
# ─────────────────────────────────────────────────────────────
# Model has a max input length (usually 512 tokens for BERT-family).
# We need to know if we'll be truncating a lot of comments.

train_df["comment_length"] = train_df["comment_text"].str.len()
print("\n=== COMMENT LENGTH (in characters) ===")
print(train_df["comment_length"].describe().round(0))

# ─────────────────────────────────────────────────────────────
# STEP 6: Save a plot for the README/blog post later
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: how frequent each toxic label is
counts = [train_df[col].sum() for col in label_cols]
axes[0].bar(label_cols, counts, color="steelblue")
axes[0].set_title("Frequency of Each Toxic Label")
axes[0].set_ylabel("Number of Comments")
axes[0].tick_params(axis="x", rotation=45)

# Right plot: distribution of comment lengths (clipped to 2000)
lengths = train_df["comment_length"].clip(upper=2000)
axes[1].hist(lengths, bins=50, color="coral", edgecolor="black")
axes[1].set_title("Comment Length Distribution")
axes[1].set_xlabel("Characters (clipped at 2000)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("data/exploration.png", dpi=100, bbox_inches="tight")
print("\n✓ Saved exploration plot to data/exploration.png")

print("\n─────────────────────────────────────────")
print("Day 1 exploration complete!")
print("─────────────────────────────────────────")
print(f"Key numbers to remember:")
print(f"  • {len(train_df):,} training comments")
print(f"  • {overall_toxic_pct:.1f}% are toxic in some way")
print(f"  • Median comment length: {int(train_df['comment_length'].median())} chars")