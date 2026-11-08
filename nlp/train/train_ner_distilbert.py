"""Fine-tune DistilBERT for WinPilot BIO entity recognition."""

import argparse
import json
import random
import re
from contextlib import nullcontext
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm


# =========================================================
# Configuration
# =========================================================

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# Set to 1 when running on a Google Colab T4; leave at 0 for the local PC.
USE_COLAB_T4 = 0
if USE_COLAB_T4 not in (0, 1):
    raise ValueError("USE_COLAB_T4 must be 0 (local PC) or 1 (Google Colab T4)")

if USE_COLAB_T4 == 1:
    PROFILE_NAME = "Google Colab T4"
    EPOCHS = 5
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 1
    FREEZE_LAYERS = 0
    USE_MIXED_PRECISION = True
else:
    PROFILE_NAME = "Local PC"
    EPOCHS = 3
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 8
    FREEZE_LAYERS = 2
    USE_MIXED_PRECISION = False

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 2
MAX_LENGTH = 256
BASE_MODEL = "distilbert/distilbert-base-uncased"

BASE_DIR = Path(__file__).resolve().parents[1]
BIO_DIR = BASE_DIR / "data" / "entities" / "augmented v2" / "bio"
MODEL_DIR = BASE_DIR / "models" / "ner" / "distilbert"
RESULTS_DIR = BASE_DIR / "results" / "ner"
RESULTS_PATH = RESULTS_DIR / "distilbert-ner.txt"

PAD_LABEL_ID = -100
ENTITY_TYPES = ("FILE", "FOLDER", "APP", "PATH_DIR", "PATH_FILE")
SECTION_WIDTH = 60


# =========================================================
# Console formatting
# =========================================================

def print_section(title):
    """Print a consistent, readable section heading for the training log."""
    print()
    print("=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


# =========================================================
# Load BIO data
# =========================================================

def load_bio_file(path):
    sentences, labels = [], []
    words, tags = [], []

    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Malformed BIO row in {path}: {line!r}")
            word, tag = parts
            words.append(word)
            tags.append(tag)

    if words:
        sentences.append(words)
        labels.append(tags)
    return sentences, labels


NER_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "validation": ("validation", "validate", "val", "dev"),
    "test": ("test", "testing"),
}


def find_ner_splits(dataset_dir):
    anchors = (dataset_dir, dataset_dir / "split", dataset_dir / "splits")
    for anchor in anchors:
        if not anchor.is_dir():
            continue

        found = {}
        children = {child.name.lower(): child for child in anchor.iterdir()}
        for split_name, aliases in NER_SPLIT_ALIASES.items():
            for alias in aliases:
                child = children.get(alias)
                if child is not None and (child.is_dir() or child.suffix.lower() in {".bio", ".jsonl"}):
                    found[split_name] = child
                    break
                for suffix in (".bio", ".jsonl"):
                    split_file = anchor / f"{alias}{suffix}"
                    if split_file.is_file():
                        found[split_name] = split_file
                        break
                if split_name in found:
                    break

        if found:
            missing = set(NER_SPLIT_ALIASES) - set(found)
            if missing:
                raise ValueError(
                    f"Found fixed NER splits in {anchor}, but missing: "
                    f"{', '.join(sorted(missing))}. Provide train, validation (or validate), and test."
                )
            return found
    return None


def load_jsonl_file(path):
    examples = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error}") from error

            text = record.get("text") if isinstance(record, dict) else None
            entities = record.get("entities") if isinstance(record, dict) else None
            if not isinstance(text, str) or not isinstance(entities, list):
                raise ValueError(
                    f"Expected text and entities fields in {path}, line {line_number}"
                )

            word_matches = list(re.finditer(r"\S+", text))
            words = [match.group(0) for match in word_matches]
            tags = ["O"] * len(words)
            for entity in entities:
                if not isinstance(entity, dict):
                    raise ValueError(f"Invalid entity in {path}, line {line_number}")
                start, end, label = entity.get("start"), entity.get("end"), entity.get("label")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not isinstance(label, str)
                    or not 0 <= start < end <= len(text)
                    or label not in ENTITY_TYPES
                ):
                    raise ValueError(
                        f"Invalid entity span or label in {path}, line {line_number}: {entity!r}"
                    )
                overlapping_words = [
                    index
                    for index, match in enumerate(word_matches)
                    if match.start() < end and match.end() > start
                ]
                if not overlapping_words:
                    raise ValueError(f"Entity does not overlap a token in {path}, line {line_number}")
                for position, word_index in enumerate(overlapping_words):
                    if tags[word_index] != "O":
                        raise ValueError(f"Overlapping entities in {path}, line {line_number}")
                    tags[word_index] = f"B-{label}" if position == 0 else f"I-{label}"
            if words:
                examples.append((words, tags))
    return examples


def load_ner_source(source):
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(source.rglob("*.bio")) + sorted(source.rglob("*.jsonl"))
    if not paths:
        return []

    examples = []
    for path in paths:
        if path.suffix.lower() == ".bio":
            words, tags = load_bio_file(path)
            examples.extend(zip(words, tags))
            print(f"{path.name:<30} {len(words):>6} sentences")
        elif path.suffix.lower() == ".jsonl":
            loaded = load_jsonl_file(path)
            examples.extend(loaded)
            print(f"{path.name:<30} {len(loaded):>6} sentences")
        else:
            raise ValueError(f"Unsupported NER split file: {path}")
    return examples


def deduplicate_ner_examples(examples, split_name):
    unique, seen = [], {}
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


def validate_ner_splits(splits):
    seen = {}
    for split_name, examples in splits.items():
        if not examples:
            raise ValueError(f"The {split_name} NER split is empty")
        for words, _ in examples:
            key = tuple(word.casefold() for word in words)
            previous_split = seen.get(key)
            if previous_split is not None and previous_split != split_name:
                raise ValueError(
                    f"Exact sentence leakage across {previous_split} and {split_name}: {' '.join(words)!r}"
                )
            seen[key] = split_name

    train_tags = {tag for _, tags in splits["train"] for tag in tags}
    evaluation_tags = {
        tag
        for split_name in ("validation", "test")
        for _, tags in splits[split_name]
        for tag in tags
    }
    unseen_tags = evaluation_tags - train_tags
    if unseen_tags:
        raise ValueError(
            "BIO tags occur in validation/test but not in train: "
            f"{', '.join(sorted(unseen_tags))}"
        )


def load_dataset(dataset_dir=BIO_DIR):
    dataset_dir = Path(dataset_dir)
    fixed_sources = find_ner_splits(dataset_dir)
    print_section("BIO FILES")

    if fixed_sources:
        splits = {
            split_name: deduplicate_ner_examples(
                load_ner_source(source), split_name
            )
            for split_name, source in fixed_sources.items()
        }
        validate_ner_splits(splits)
        print_section("FIXED DATA SPLITS")
        print(f"Source     : {dataset_dir}")
        print(f"Training   : {len(splits['train'])}")
        print(f"Validation : {len(splits['validation'])}")
        print(f"Testing    : {len(splits['test'])}")
        return splits["train"], splits["validation"], splits["test"]

    examples = load_ner_source(dataset_dir)
    if not examples:
        raise FileNotFoundError(f"No .bio or compatible .jsonl files found in {dataset_dir}")
    unique = deduplicate_ner_examples(examples, "full dataset")

    print_section("DATASET")
    print(f"Total sentences     : {len(examples)}")
    print(f"Duplicates removed  : {len(examples) - len(unique)}")
    print(f"Unique sentences    : {len(unique)}")

    if len(unique) < 3:
        raise ValueError(f"Expected at least 3 unique NER examples in {dataset_dir}")

    random.Random(SEED).shuffle(unique)
    train_end = int(len(unique) * TRAIN_RATIO)
    val_end = train_end + int(len(unique) * VAL_RATIO)
    splits = {
        "train": unique[:train_end],
        "validation": unique[train_end:val_end],
        "test": unique[val_end:],
    }
    validate_ner_splits(splits)
    print_section("AUTOMATIC DATA SPLIT")
    print(f"Source     : {dataset_dir}")
    print(f"Training   : {len(splits['train'])}")
    print(f"Validation : {len(splits['validation'])}")
    print(f"Testing    : {len(splits['test'])}")
    return splits["train"], splits["validation"], splits["test"]


# =========================================================
# Entity metrics
# =========================================================

def get_entities(tags):
    entities = set()
    active_type = None
    start = None

    for index, tag in enumerate(tags):
        if tag.startswith("B-"):
            if active_type is not None:
                entities.add((active_type, start, index - 1))
            active_type = tag[2:]
            start = index
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if active_type != entity_type:
                if active_type is not None:
                    entities.add((active_type, start, index - 1))
                active_type = entity_type
                start = index
        else:
            if active_type is not None:
                entities.add((active_type, start, index - 1))
            active_type = None
            start = None

    if active_type is not None:
        entities.add((active_type, start, len(tags) - 1))
    return entities


def entity_metrics(gold_sequences, predicted_sequences):
    totals = {kind: [0, 0, 0] for kind in ENTITY_TYPES}
    for gold, predicted in zip(gold_sequences, predicted_sequences):
        true_entities = get_entities(gold)
        pred_entities = get_entities(predicted)
        for kind in ENTITY_TYPES:
            true_set = {entity for entity in true_entities if entity[0] == kind}
            pred_set = {entity for entity in pred_entities if entity[0] == kind}
            totals[kind][0] += len(true_set & pred_set)
            totals[kind][1] += len(pred_set - true_set)
            totals[kind][2] += len(true_set - pred_set)

    metrics = {}
    for kind, (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[kind] = {"precision": precision, "recall": recall, "f1": f1,
                         "tp": tp, "fp": fp, "fn": fn}
    return metrics


# =========================================================
# Tokenization and label alignment
# =========================================================

def encode_examples(examples, tokenizer, tag_to_id):
    encoded_examples = []
    for words, tags in examples:
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        aligned_labels = []
        word_positions = []
        previous_word_id = None
        for position, word_id in enumerate(encoded.word_ids()):
            if word_id is None:
                aligned_labels.append(PAD_LABEL_ID)
            elif word_id != previous_word_id:
                aligned_labels.append(tag_to_id[tags[word_id]])
                word_positions.append(position)
            else:
                aligned_labels.append(PAD_LABEL_ID)
            previous_word_id = word_id

        retained_words = len(word_positions)
        encoded_examples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": aligned_labels,
            "word_positions": word_positions,
            "gold_tags": tags[:retained_words],
            "words": words[:retained_words],
        })
    return encoded_examples


def make_collator(tokenizer):
    def collate(batch):
        model_inputs = tokenizer.pad(
            [{"input_ids": item["input_ids"],
              "attention_mask": item["attention_mask"]} for item in batch],
            padding=True,
            return_tensors="pt",
        )
        max_length = model_inputs["input_ids"].shape[1]
        labels = torch.full((len(batch), max_length), PAD_LABEL_ID, dtype=torch.long)
        for row, item in enumerate(batch):
            labels[row, :len(item["labels"])] = torch.tensor(item["labels"], dtype=torch.long)
        model_inputs["labels"] = labels
        metadata = {
            "word_positions": [item["word_positions"] for item in batch],
            "gold_tags": [item["gold_tags"] for item in batch],
            "words": [item["words"] for item in batch],
        }
        return model_inputs, metadata
    return collate


# =========================================================
# Evaluation and report writing
# =========================================================

def evaluate(model, loader, device, id_to_tag):
    model.eval()
    gold_sequences, predicted_sequences = [], []
    with torch.inference_mode():
        for inputs, metadata in tqdm(loader, desc="Evaluating", leave=False):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
            for row, positions in enumerate(metadata["word_positions"]):
                predicted = [id_to_tag[logits[row][position]] for position in positions]
                gold = metadata["gold_tags"][row]
                gold_sequences.append(gold)
                predicted_sequences.append(predicted)
    return gold_sequences, predicted_sequences, entity_metrics(gold_sequences, predicted_sequences)


def write_results(path, title, gold_sequences, predicted_sequences, metrics):
    labels = sorted(
        {tag for sequence in gold_sequences for tag in sequence}
        | {tag for sequence in predicted_sequences for tag in sequence}
    )
    all_gold = [tag for sequence in gold_sequences for tag in sequence]
    all_predicted = [tag for sequence in predicted_sequences for tag in sequence]
    lines = [title, "=" * len(title), "", "TOKEN-LEVEL EVALUATION",
             classification_report(all_gold, all_predicted, labels=labels, zero_division=0),
             "", "ENTITY-LEVEL EVALUATION"]
    for kind, result in metrics.items():
        lines.extend([
            f"{kind}: precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} f1={result['f1']:.4f} "
            f"TP={result['tp']} FP={result['fp']} FN={result['fn']}"
        ])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================
# Main training pipeline
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for WinPilot named entity recognition."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=BIO_DIR,
        help="BIO/JSONL root; uses fixed train/validation/test splits when available.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=FREEZE_LAYERS,
        help="Freeze this many lower DistilBERT transformer layers (quick CPU runs).",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch-size, and gradient-accumulation-steps must be positive")
    if args.freeze_layers < 0:
        parser.error("freeze-layers cannot be negative")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_COLAB_T4 == 1 and device.type != "cuda":
        raise RuntimeError(
            "The Colab T4 profile is enabled, but CUDA is unavailable. "
            "In Colab, select Runtime > Change runtime type > T4 GPU."
        )
    use_mixed_precision = USE_MIXED_PRECISION and device.type == "cuda"
    train_data, val_data, test_data = load_dataset(args.dataset_dir)
    print_section("DEVICE")
    print(f"Profile             : {PROFILE_NAME}")
    print(f"Device              : {device}")
    if device.type == "cuda":
        print(f"GPU                 : {torch.cuda.get_device_name(0)}")
    print(f"Pretrained model    : {BASE_MODEL}")
    print(f"Epochs              : {args.epochs}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Frozen layers       : {args.freeze_layers}")
    print(f"Mixed precision     : {'FP16' if use_mixed_precision else 'off'}")
    print(f"Learning rate       : {LEARNING_RATE:g}")
    print(f"Early-stop patience : {PATIENCE}", flush=True)

    all_tags = sorted({tag for _, tags in train_data for tag in tags})
    evaluation_tags = {
        tag
        for _, tags in val_data + test_data
        for tag in tags
    }
    unseen_tags = evaluation_tags - set(all_tags)
    if unseen_tags:
        raise ValueError(
            "Validation/test contain BIO tags missing from the training split: "
            f"{', '.join(sorted(unseen_tags))}"
        )
    tag_to_id = {tag: index for index, tag in enumerate(all_tags)}
    id_to_tag = {index: tag for tag, index in tag_to_id.items()}

    print_section("LABELS")
    print(f"Number of tags      : {len(all_tags)}")
    print(f"Tag set             : {all_tags}")

    print_section("MODEL")
    print("Loading tokenizer and pretrained weights...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(all_tags),
        id2label=id_to_tag,
        label2id=tag_to_id,
    ).to(device)

    transformer_layers = model.distilbert.transformer.layer
    if args.freeze_layers > len(transformer_layers):
        parser.error(f"freeze-layers cannot exceed {len(transformer_layers)}")
    if args.freeze_layers:
        for parameter in model.distilbert.embeddings.parameters():
            parameter.requires_grad = False
        for layer in transformer_layers[:args.freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        print(
            f"Frozen embeddings and first {args.freeze_layers} transformer layers.",
            flush=True,
        )
    else:
        print("All transformer layers are trainable.", flush=True)

    collator = make_collator(tokenizer)
    print("Tokenizing BIO sentences...", flush=True)
    train_encoded = encode_examples(train_data, tokenizer, tag_to_id)
    val_encoded = encode_examples(val_data, tokenizer, tag_to_id)
    test_encoded = encode_examples(test_data, tokenizer, tag_to_id)
    print("Tokenization complete.", flush=True)
    train_loader = DataLoader(
        train_encoded,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_encoded,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_encoded,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print_section("BATCHING")
    print(f"Training batches    : {len(train_loader)}")
    print(f"Validation batches  : {len(val_loader)}")
    print(f"Testing batches     : {len(test_loader)}")

    trainable_parameters = (
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    optimizer_steps_per_epoch = (
        len(train_loader) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    total_steps = args.epochs * optimizer_steps_per_epoch
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / warmup_steps,
            max(
                0.0,
                (total_steps - step) / max(1, total_steps - warmup_steps),
            ),
        ),
    )

    best_path_file_f1 = -1.0
    best_overall_f1 = -1.0
    epochs_without_improvement = 0
    print_section("TRAINING")
    print("Starting fine-tuning...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, (inputs, _) in enumerate(progress, start=1):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            amp_context = (
                torch.autocast("cuda", dtype=torch.float16)
                if use_mixed_precision
                else nullcontext()
            )
            with amp_context:
                loss = model(**inputs).loss
            group_start = (
                (step - 1) // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps,
                len(train_loader) - group_start,
            )
            scaler.scale(loss / group_size).backward()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler skips optimizer.step() when FP16 gradients overflow.
                # Do not advance the LR schedule for a skipped optimizer update.
                if scaler.get_scale() >= scale_before_step:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_gold, val_pred, val_metrics = evaluate(model, val_loader, device, id_to_tag)
        path_file_f1 = val_metrics["PATH_FILE"]["f1"]
        overall_f1 = (
            sum(item["f1"] for item in val_metrics.values())
            / len(val_metrics)
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {running_loss / max(1, len(train_loader)):.4f} | "
            f"Val PATH_FILE F1: {path_file_f1:.4f} | "
            f"Val Macro Entity F1: {overall_f1:.4f}",
            flush=True,
        )

        improved = overall_f1 > best_overall_f1

        if improved:
            best_overall_f1 = overall_f1
            epochs_without_improvement = 0

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            
            (MODEL_DIR / "entity_labels.json").write_text(
                json.dumps({
                    "tag_to_id": tag_to_id, 
                    "id_to_tag": id_to_tag
                }, indent=2),
                encoding="utf-8",
            )
            print("Best model saved.", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"No improvement: {epochs_without_improvement}/{PATIENCE}", flush=True)
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping.", flush=True)
                break

    print_section("TEST EVALUATION")
    print("Loading best saved checkpoint...", flush=True)
    best_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    test_gold, test_pred, test_metrics = evaluate(
        best_model,
        test_loader,
        device,
        id_to_tag,
    )
    write_results(
        RESULTS_PATH,
        "DistilBERT NER Test Results",
        test_gold,
        test_pred,
        test_metrics,
    )

    print_section("TOKEN-LEVEL EVALUATION")
    all_gold = [tag for sequence in test_gold for tag in sequence]
    all_predicted = [tag for sequence in test_pred for tag in sequence]
    labels = sorted(set(all_gold) | set(all_predicted))
    print(classification_report(
        all_gold,
        all_predicted,
        labels=labels,
        zero_division=0,
    ))

    print_section("ENTITY-LEVEL EVALUATION")
    total_tp = total_fp = total_fn = 0
    for kind, result in test_metrics.items():
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        print()
        print(kind)
        print(f"  Precision : {result['precision']:.4f}")
        print(f"  Recall    : {result['recall']:.4f}")
        print(f"  F1        : {result['f1']:.4f}")
        print(f"  TP        : {result['tp']}")
        print(f"  FP        : {result['fp']}")
        print(f"  FN        : {result['fn']}")

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    print()
    print("OVERALL ENTITY PERFORMANCE")
    print(f"Precision : {micro_precision:.4f}")
    print(f"Recall    : {micro_recall:.4f}")
    print(f"F1        : {micro_f1:.4f}")
    print(f"TP        : {total_tp}")
    print(f"FP        : {total_fp}")
    print(f"FN        : {total_fn}")
    print(
        "Macro F1  : "
        f"{sum(item['f1'] for item in test_metrics.values()) / len(test_metrics):.4f}"
    )
    print_section("MODEL SAVED")
    print(f"Model directory     : {MODEL_DIR}")
    print(f"Test report         : {RESULTS_PATH}")
    print(f"Test PATH_FILE F1   : {test_metrics['PATH_FILE']['f1']:.4f}")


if __name__ == "__main__":
    main()
