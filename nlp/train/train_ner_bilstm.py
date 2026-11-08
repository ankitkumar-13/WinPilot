
import argparse
import random
import csv
import json

import torch
import torch.nn as nn

from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence
)
from tqdm import tqdm


# =========================================================
# Configuration
# =========================================================

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 3

EMBEDDING_DIM = 100
HIDDEN_SIZE = 128

PAD_TAG = -100


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

BIO_DIR = (
    BASE_DIR
    / "data"
    / "entities"
    / "augmented v2"
    / "bio"
)

parser = argparse.ArgumentParser(description="Train the WinPilot BiLSTM NER model.")
parser.add_argument(
    "--dataset-dir",
    type=Path,
    default=BIO_DIR,
    help="BIO split root; uses fixed train/validation/test splits when available.",
)
args = parser.parse_args()

MODEL = (
    BASE_DIR
    / "models"
    / "ner"
    / "ner_bilstm"
    / "ner_model_v2.pt"
)


# =========================================================
# Reproducibility
# =========================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)
print(DEVICE)

if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))


# =========================================================
# Load BIO file
# =========================================================

def load_data(path):
    sentences = []
    tags = []

    words = []
    labs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labs)

                    words = []
                    labs = []

                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid BIO line in {path}: {line!r}"
                )

            word, tag = parts

            words.append(word.lower())
            labs.append(tag)

    if words:
        sentences.append(words)
        tags.append(labs)

    return sentences, tags


# =========================================================
# Load fixed splits or all BIO files
# =========================================================

def find_bio_splits(dataset_dir):
    aliases = {
        "train": ("train", "training"),
        "validation": ("validation", "validate", "val", "dev"),
        "test": ("test", "testing"),
    }
    for anchor in (dataset_dir, dataset_dir / "split", dataset_dir / "splits"):
        if not anchor.is_dir():
            continue
        children = {child.name.lower(): child for child in anchor.iterdir()}
        sources = {}
        for split_name, split_aliases in aliases.items():
            for alias in split_aliases:
                child = children.get(alias)
                if child is not None and (child.is_dir() or child.suffix.lower() == ".bio"):
                    sources[split_name] = child
                    break
                split_file = anchor / f"{alias}.bio"
                if split_file.is_file():
                    sources[split_name] = split_file
                    break
        if sources:
            missing = set(aliases) - set(sources)
            if missing:
                raise ValueError(
                    f"Found fixed BIO splits in {anchor}, but missing: "
                    f"{', '.join(sorted(missing))}. Provide train, validation (or validate), and test."
                )
            return sources
    return None


def load_bio_source(source):
    files = [source] if source.is_file() else sorted(source.rglob("*.bio"))
    examples = []
    for path in files:
        x, y = load_data(path)
        examples.extend(zip(x, y))
        print(f"{path.name:<30} {len(x):>6} sentences")
    return examples


def clean_bio_split(examples, split_name):
    unique = []
    seen = {}
    for words, tags in examples:
        key = tuple(word.casefold() for word in words)
        previous_tags = seen.get(key)
        if previous_tags is not None:
            if previous_tags != tuple(tags):
                raise ValueError(
                    f"Conflicting BIO labels for duplicate {split_name} sentence: {' '.join(words)!r}"
                )
            continue
        seen[key] = tuple(tags)
        unique.append((words, tags))
    return unique


def check_bio_split_leakage(splits):
    seen = {}
    for split_name, examples in splits.items():
        if not examples:
            raise ValueError(f"The {split_name} NER split is empty")
        for words, _ in examples:
            key = tuple(word.casefold() for word in words)
            previous = seen.get(key)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"Exact sentence leakage across {previous} and {split_name}: {' '.join(words)!r}"
                )
            seen[key] = split_name


print()
print("=" * 60)
print("BIO FILES")
print("=" * 60)

fixed_sources = find_bio_splits(args.dataset_dir)
if fixed_sources:
    fixed_splits = {
        split_name: clean_bio_split(load_bio_source(source), split_name)
        for split_name, source in fixed_sources.items()
    }
    check_bio_split_leakage(fixed_splits)
    train_x, train_y = zip(*fixed_splits["train"])
    val_x, val_y = zip(*fixed_splits["validation"])
    test_x, test_y = zip(*fixed_splits["test"])
    train_x, train_y = list(train_x), list(train_y)
    val_x, val_y = list(val_x), list(val_y)
    test_x, test_y = list(test_x), list(test_y)
    split_source = "fixed"
else:
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"BIO dataset directory not found: {args.dataset_dir}")
    data = load_bio_source(args.dataset_dir)
    if not data:
        raise FileNotFoundError(f"No .bio files found in {args.dataset_dir}")
    unique = clean_bio_split(data, "full dataset")
    print(f"Total sentences     : {len(data)}")
    print(f"Duplicates removed  : {len(data) - len(unique)}")
    print(f"Unique sentences    : {len(unique)}")
    if len(unique) < 3:
        raise ValueError("At least 3 unique sentences are required.")

    random.shuffle(unique)
    all_x, all_y = zip(*unique)
    all_x, all_y = list(all_x), list(all_y)
    total = len(all_x)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    train_x, train_y = all_x[:train_end], all_y[:train_end]
    val_x, val_y = all_x[train_end:val_end], all_y[train_end:val_end]
    test_x, test_y = all_x[val_end:], all_y[val_end:]
    split_source = "automatic 80/10/10"

splits = {
    "train": list(zip(train_x, train_y)),
    "validation": list(zip(val_x, val_y)),
    "test": list(zip(test_x, test_y)),
}
check_bio_split_leakage(splits)

all_x = train_x + val_x + test_x
all_y = train_y + val_y + test_y
print()
print("=" * 60)
print("DATA SPLIT")
print("=" * 60)
print(f"Source     : {split_source}")
print(f"Training   : {len(train_x)}")
print(f"Validation : {len(val_x)}")
print(f"Testing    : {len(test_x)}")

if not train_x or not val_x or not test_x:
    raise ValueError(
        "Train, validation and test splits must all be non-empty."
    )

# =========================================================
# Build vocabulary from training data only
# =========================================================

word_count = Counter(
    word
    for sentence in train_x
    for word in sentence
)

word2id = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word in word_count:
    word2id[word] = len(word2id)


# =========================================================
# Build tag dictionary
# =========================================================

all_tags = sorted({
    tag
    for sentence_tags in train_y
    for tag in sentence_tags
})
evaluation_tags = {
    tag
    for sentence_tags in val_y + test_y
    for tag in sentence_tags
}
unseen_tags = evaluation_tags - set(all_tags)
if unseen_tags:
    raise ValueError(
        "Validation/test contain BIO tags missing from the training split: "
        f"{', '.join(sorted(unseen_tags))}"
    )

tag2id = {
    tag: i
    for i, tag in enumerate(all_tags)
}

id2tag = {
    i: tag
    for tag, i in tag2id.items()
}

print()
print("=" * 60)
print("VOCABULARY")
print("=" * 60)

print(f"Vocabulary : {len(word2id)}")
print(f"Tags       : {len(tag2id)}")
print(f"Tag set    : {all_tags}")


# =========================================================
# Encode data
# =========================================================

def encode_x(sentence):
    return [
        word2id.get(word, word2id["<UNK>"])
        for word in sentence
    ]


def encode_y(tags):
    return [tag2id[tag] for tag in tags]


# Preserve original test tokens for readable error analysis.
test_words = [sentence.copy() for sentence in test_x]

train_x = [encode_x(x) for x in train_x]
train_y = [encode_y(y) for y in train_y]

val_x = [encode_x(x) for x in val_x]
val_y = [encode_y(y) for y in val_y]

test_x = [encode_x(x) for x in test_x]
test_y = [encode_y(y) for y in test_y]


# =========================================================
# Batched data loading
# =========================================================

def collate_fn(batch):
    xs, ys = zip(*batch)

    lengths = torch.tensor(
        [len(x) for x in xs],
        dtype=torch.long
    )

    xs = pad_sequence(
        [
            torch.tensor(x, dtype=torch.long)
            for x in xs
        ],
        batch_first=True,
        padding_value=0
    )

    ys = pad_sequence(
        [
            torch.tensor(y, dtype=torch.long)
            for y in ys
        ],
        batch_first=True,
        padding_value=PAD_TAG
    )

    return xs, ys, lengths


train_loader = DataLoader(
    list(zip(train_x, train_y)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=DEVICE.type == "cuda",
    num_workers=0
)

val_loader = DataLoader(
    list(zip(val_x, val_y)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=DEVICE.type == "cuda",
    num_workers=0
)

test_loader = DataLoader(
    list(zip(test_x, test_y)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=DEVICE.type == "cuda",
    num_workers=0
)

print()
print("=" * 60)
print("BATCHING")
print("=" * 60)

print(f"Batch size          : {BATCH_SIZE}")
print(f"Training batches    : {len(train_loader)}")
print(f"Validation batches  : {len(val_loader)}")
print(f"Testing batches     : {len(test_loader)}")


# =========================================================
# Model
# =========================================================

class NERModel(nn.Module):

    def __init__(self, vocab_size, tag_count):
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE * 2,
            tag_count
        )

    def forward(self, x, lengths):
        x = self.emb(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1)
        )

        return self.fc(out)


model = NERModel(
    len(word2id),
    len(tag2id)
).to(DEVICE)


# =========================================================
# Loss and optimizer
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_fn = nn.CrossEntropyLoss(
    ignore_index=PAD_TAG
)


# =========================================================
# Validation loss
# =========================================================

def get_loss(loader):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for x, y, lengths in loader:
            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            out = model(x, lengths)

            loss = loss_fn(
                out.reshape(-1, len(tag2id)),
                y.reshape(-1)
            )

            count = (y != PAD_TAG).sum().item()

            total_loss += loss.item() * count
            total_tokens += count

    return total_loss / max(total_tokens, 1)


# =========================================================
# Training with early stopping
# =========================================================

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

best_val_loss = float("inf")
bad_epochs = 0

MODEL.parent.mkdir(
    parents=True,
    exist_ok=True
)

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for x, y, lengths in progress:
        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        out = model(x, lengths)

        loss = loss_fn(
            out.reshape(-1, len(tag2id)),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        count = (y != PAD_TAG).sum().item()

        total_loss += loss.item() * count
        total_tokens += count

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = (
        total_loss / max(total_tokens, 1)
    )

    val_loss = get_loss(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save(
            {
                "model": model.state_dict(),
                "word2id": word2id,
                "tag2id": tag2id,
                "config": {
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_size": HIDDEN_SIZE,
                    "batch_size": BATCH_SIZE
                }
            },
            MODEL
        )

        print("Best model saved.")

    else:
        bad_epochs += 1

        print(
            f"No improvement: "
            f"{bad_epochs}/{PATIENCE}"
        )

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break


# =========================================================
# Load best model
# =========================================================

checkpoint = torch.load(
    MODEL,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model"]
)

word2id = checkpoint["word2id"]
tag2id = checkpoint["tag2id"]

id2tag = {
    i: tag
    for tag, i in tag2id.items()
}

model.eval()


# =========================================================
# Entity extraction
# =========================================================

def extract_entities(tags):
    entities = []

    current = None
    start = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current is not None:
                entities.append(
                    (current, start, i - 1)
                )

            current = tag[2:]
            start = i

        elif tag.startswith("I-"):
            typ = tag[2:]

            if current != typ:
                if current is not None:
                    entities.append(
                        (current, start, i - 1)
                    )

                current = typ
                start = i

        else:
            if current is not None:
                entities.append(
                    (current, start, i - 1)
                )

            current = None
            start = None

    if current is not None:
        entities.append(
            (current, start, len(tags) - 1)
        )

    return set(entities)


# =========================================================
# PATH_FILE error inspection
# =========================================================

def get_path_file_error(tokens, true_tags, pred_tags, sentence_id):
    true_ents = {
        e for e in extract_entities(true_tags)
        if e[0] == "PATH_FILE"
    }

    pred_ents = {
        e for e in extract_entities(pred_tags)
        if e[0] == "PATH_FILE"
    }

    if true_ents == pred_ents:
        return None

    def entity_texts(entities):
        return [
            {
                "text": " ".join(tokens[start:end + 1]),
                "start_token": start,
                "end_token": end
            }
            for _, start, end in sorted(entities)
        ]

    missing = true_ents - pred_ents
    extra = pred_ents - true_ents

    mismatches = [
        {
            "token_index": i,
            "token": token,
            "expected_tag": true_tag,
            "predicted_tag": pred_tag
        }
        for i, (token, true_tag, pred_tag) in enumerate(
            zip(tokens, true_tags, pred_tags)
        )
        if true_tag != pred_tag
    ]

    return {
        "sentence_id": sentence_id,
        "command": " ".join(tokens),
        "expected_path_file": json.dumps(
            entity_texts(true_ents), ensure_ascii=False
        ),
        "predicted_path_file": json.dumps(
            entity_texts(pred_ents), ensure_ascii=False
        ),
        "missing_path_file": json.dumps(
            entity_texts(missing), ensure_ascii=False
        ),
        "extra_path_file": json.dumps(
            entity_texts(extra), ensure_ascii=False
        ),
        "token_tag_mismatches": json.dumps(
            mismatches, ensure_ascii=False
        )
    }


# =========================================================
# Batched test evaluation
# =========================================================

all_true = []
all_pred = []

entity_stats = {
    "FILE": [0, 0, 0],
    "FOLDER": [0, 0, 0],
    "APP": [0, 0, 0],
    "PATH_DIR": [0, 0, 0],
    "PATH_FILE": [0, 0, 0]
}

sentence_index = 0
path_file_errors = []

with torch.inference_mode():
    for x, y, lengths in tqdm(
        test_loader,
        desc="Testing"
    ):
        x = x.to(
            DEVICE,
            non_blocking=True
        )

        out = model(x, lengths)

        predictions = out.argmax(dim=2).cpu().tolist()
        true_batch = y.tolist()

        for true_ids, pred_ids, length in zip(
            true_batch,
            predictions,
            lengths.tolist()
        ):
            true_names = [
                id2tag[v]
                for v in true_ids[:length]
            ]

            pred_names = [
                id2tag[v]
                for v in pred_ids[:length]
            ]

            error = get_path_file_error(
                test_words[sentence_index],
                true_names,
                pred_names,
                sentence_index
            )

            if error is not None:
                path_file_errors.append(error)

            sentence_index += 1

            all_true.extend(true_names)
            all_pred.extend(pred_names)

            true_entities = extract_entities(
                true_names
            )

            pred_entities = extract_entities(
                pred_names
            )

            for typ in entity_stats:
                true_set = {
                    e for e in true_entities
                    if e[0] == typ
                }

                pred_set = {
                    e for e in pred_entities
                    if e[0] == typ
                }

                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)

                entity_stats[typ][0] += tp
                entity_stats[typ][1] += fp
                entity_stats[typ][2] += fn


# =========================================================
# Export PATH_FILE errors to CSV
# =========================================================

ERROR_REPORT = MODEL.parent / "path_file_errors.csv"

error_columns = [
    "sentence_id",
    "command",
    "expected_path_file",
    "predicted_path_file",
    "missing_path_file",
    "extra_path_file",
    "token_tag_mismatches"
]

with open(ERROR_REPORT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=error_columns)
    writer.writeheader()
    writer.writerows(path_file_errors)

print()
print("=" * 60)
print("PATH_FILE ERROR REPORT")
print("=" * 60)
print(f"Sentences with PATH_FILE errors : {len(path_file_errors)}")
print(f"CSV report saved to             : {ERROR_REPORT}")


# =========================================================
# Token-level evaluation
# =========================================================

print()
print("=" * 60)
print("TOKEN-LEVEL EVALUATION")
print("=" * 60)

print(
    classification_report(
        all_true,
        all_pred,
        labels=sorted(id2tag.values()),
        zero_division=0
    )
)


# =========================================================
# Entity-level evaluation
# =========================================================

print()
print("=" * 60)
print("ENTITY-LEVEL EVALUATION")
print("=" * 60)

all_tp = 0
all_fp = 0
all_fn = 0

for typ, values in entity_stats.items():
    tp, fp, fn = values

    all_tp += tp
    all_fp += fp
    all_fn += fn

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print()
    print(typ)
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP        : {tp}")
    print(f"  FP        : {fp}")
    print(f"  FN        : {fn}")


# =========================================================
# Overall entity performance
# =========================================================

precision = (
    all_tp / (all_tp + all_fp)
    if all_tp + all_fp
    else 0
)

recall = (
    all_tp / (all_tp + all_fn)
    if all_tp + all_fn
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall
    else 0
)

print()
print("=" * 60)
print("OVERALL ENTITY PERFORMANCE")
print("=" * 60)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")

print(f"TP        : {all_tp}")
print(f"FP        : {all_fp}")
print(f"FN        : {all_fn}")


# =========================================================
# Final
# =========================================================

print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(MODEL)
