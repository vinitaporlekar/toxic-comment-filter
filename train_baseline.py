import json
import time
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

Path("models").mkdir(exist_ok=True)

# STEP 1: Load data
print("Loading dataset...")
ds = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")
df = ds["train"].to_pandas()

label_cols = ["toxic", "severe_toxic", "obscene", "threat",
              "insult", "identity_hate"]
df["any_toxic"] = df[label_cols].max(axis=1)

X = df["comment_text"].astype(str).to_numpy()
y = df["any_toxic"].astype(int).to_numpy()
print(f"Total: {len(X):,} comments, {y.mean() * 100:.1f}% toxic")

# STEP 2: Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Validation: {len(X_val):,}")

# STEP 3: Vectorize
print("\nVectorizing text with TF-IDF...")
t0 = time.time()
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    stop_words="english"
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f"Vectorization took {time.time() - t0:.1f}s")

# STEP 4: Train
print("\nTraining logistic regression...")
t0 = time.time()
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    n_jobs=-1
)
model.fit(X_train_vec, y_train)
print(f"Training took {time.time() - t0:.1f}s")

# STEP 5: Evaluate
print("\nEvaluating on validation set...")
y_pred = model.predict(X_val_vec)

precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"\n{'─' * 50}")
print(f"  BASELINE RESULTS")
print(f"{'─' * 50}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}  ← number to beat")
print(f"{'─' * 50}")

print("\nConfusion matrix:")
print(confusion_matrix(y_val, y_pred))
print("\nDetailed report:")
print(classification_report(y_val, y_pred,
                             target_names=["not_toxic", "toxic"]))

# STEP 6: Save
joblib.dump(model, "models/baseline_model.joblib")
joblib.dump(vectorizer, "models/baseline_vectorizer.joblib")
print("\n✓ Saved model and vectorizer to models/")

results = {
    "model": "TF-IDF + Logistic Regression",
    "train_size": len(X_train),
    "val_size": len(X_val),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1": round(float(f1), 4),
}
with open("models/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✓ Saved metrics to models/baseline_results.json")

# STEP 7: Try on new examples
print(f"\n{'─' * 50}")
print("  TRYING IT ON NEW COMMENTS")
print(f"{'─' * 50}")

test_comments = [
    "You're a wonderful person, thank you!",
    "I hate you so much, you're worthless",
    "The weather looks nice today",
    "shut up you idiot nobody cares what you think",
    "Great article, I learned a lot from it",
]

for comment in test_comments:
    vec = vectorizer.transform([comment])
    prob_toxic = model.predict_proba(vec)[0][1]
    pred = "🚨 TOXIC" if prob_toxic > 0.5 else "✓ clean"
    print(f"  {pred}  ({prob_toxic:.2f})  \"{comment}\"")

print("\n─────────────────────────────────────────")
print("Day 2 complete!")
print("─────────────────────────────────────────")