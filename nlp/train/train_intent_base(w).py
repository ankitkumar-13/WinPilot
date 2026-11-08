import argparse
import pickle

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from intent_dataset_splits import load_intent_splits


# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "intent" / "augmented v2"
MODEL_DIR = BASE_DIR / "models" / "intent"
MODEL_PATH = MODEL_DIR / "base_intent_model_v2(w).pkl"

parser = argparse.ArgumentParser(description="Train the word-only WinPilot intent model.")
parser.add_argument(
    "--dataset-dir",
    type=Path,
    default=DATASET_DIR,
    help="Dataset root; uses fixed train/validation/test splits when available.",
)
args = parser.parse_args()


# =========================
# LOAD DATASET
# =========================

split_data, split_source = load_intent_splits(args.dataset_dir)
X_train = [text for text, _ in split_data["train"]]
y_train = [label for _, label in split_data["train"]]
X_val = [text for text, _ in split_data["validation"]]
y_val = [label for _, label in split_data["validation"]]
X_test = [text for text, _ in split_data["test"]]
y_test = [label for _, label in split_data["test"]]
texts = X_train + X_val + X_test
labels = y_train + y_val + y_test


# =========================
# VALIDATE DATASET
# =========================

if not X_train:
    raise ValueError(
        f"No training data found in: {args.dataset_dir}"
    )

if len(set(labels)) < 2:
    raise ValueError(
        "At least two intents are required for training."
    )


# =========================
# DISPLAY DATASET INFO
# =========================

print("=" * 60)
print("WINPILOT NLP TRAINING")
print("=" * 60)

print(f"\nTotal examples: {len(texts)}")
print(f"Intents: {len(set(labels))}")

print("\nExamples per intent:")

for intent in sorted(set(labels)):

    count = labels.count(intent)

    print(f"  {intent}: {count}")


# =========================
# DATA SPLITS
# =========================

print("\nDataset split:")
print(f"  Source:              {split_source}")
print(f"  Training examples: {len(X_train)}")
print(f"  Validation examples: {len(X_val)}")
print(f"  Testing examples:  {len(X_test)}")


# =========================
# WORD TF-IDF + LOGISTIC REGRESSION
# =========================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)


# =========================
# EVALUATION
# =========================

y_val_pred = model.predict(X_val)
y_pred = model.predict(X_test)


# =========================
# SHOW MISCLASSIFIED EXAMPLES
# =========================

print("\n" + "=" * 60)
print("MISCLASSIFIED EXAMPLES")
print("=" * 60)

misclassified = 0

for text, actual, predicted in zip(X_test, y_test, y_pred):

    if actual != predicted:

        misclassified += 1

        print(f"\nCommand:   {text}")
        print(f"Actual:    {actual}")
        print(f"Predicted: {predicted}")


if misclassified == 0:
    print("\nNo misclassified examples.")


# =========================
# MODEL EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)
validation_accuracy = accuracy_score(y_val, y_val_pred)
intent_names = sorted(set(labels))

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nValidation accuracy: {validation_accuracy:.4f}")
print("Validation report:")
print(classification_report(
    y_val,
    y_val_pred,
    labels=intent_names,
    zero_division=0,
    digits=4,
))

print(f"Test accuracy: {accuracy:.4f}")

print("Test report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=intent_names,
        zero_division=0,
        digits=4,
    )
)


# =========================
# SAVE MODEL
# =========================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)


# =========================
# COMPLETION
# =========================

print("\n" + "=" * 60)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 60)

print(f"\nModel saved to: {MODEL_PATH}")
