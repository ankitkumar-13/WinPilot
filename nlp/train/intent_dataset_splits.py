"""Load fixed intent splits or create reproducible train/validation/test splits."""

import json
import random
from collections import defaultdict
from pathlib import Path


SPLIT_ALIASES = {
    "train": ("train", "training"),
    "validation": ("validation", "validate", "val", "dev"),
    "test": ("test", "testing"),
}


def _read_jsonl(path):
    examples = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error}") from error
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Expected a text field in {path}, line {line_number}")
            if not isinstance(record.get("intent"), str) or not record["intent"].strip():
                raise ValueError(f"Expected an intent field in {path}, line {line_number}")
            examples.append((record["text"].strip(), record["intent"].strip().lower()))
    return examples


def _read_intent_source(source):
    if source.is_file():
        if source.suffix.lower() != ".jsonl":
            raise ValueError(
                f"A split file must be JSONL with text and intent fields: {source}"
            )
        return _read_jsonl(source)

    examples = []
    for path in sorted(source.rglob("*.txt")):
        intent = path.stem.lower()
        with path.open("r", encoding="utf-8") as stream:
            examples.extend(
                (line.strip(), intent)
                for line in stream
                if line.strip()
            )
    for path in sorted(source.rglob("*.jsonl")):
        examples.extend(_read_jsonl(path))
    return examples


def _find_fixed_splits(dataset_dir):
    anchors = (dataset_dir, dataset_dir / "split", dataset_dir / "splits")
    for anchor in anchors:
        if not anchor.is_dir():
            continue

        found = {}
        children = {child.name.lower(): child for child in anchor.iterdir()}
        for split_name, aliases in SPLIT_ALIASES.items():
            for alias in aliases:
                child = children.get(alias)
                if child is not None and (child.is_dir() or child.suffix.lower() == ".jsonl"):
                    found[split_name] = child
                    break
                jsonl_path = anchor / f"{alias}.jsonl"
                if jsonl_path.is_file():
                    found[split_name] = jsonl_path
                    break

        if found:
            missing = set(SPLIT_ALIASES) - set(found)
            if missing:
                raise ValueError(
                    f"Found fixed intent splits in {anchor}, but missing: "
                    f"{', '.join(sorted(missing))}. Provide train, validation (or validate), and test."
                )
            return found
    return None


def _normalize_text(text):
    return " ".join(text.casefold().split())


def _deduplicate_split(examples, split_name):
    unique = []
    labels_by_text = {}
    for text, label in examples:
        if not text:
            continue
        key = _normalize_text(text)
        previous = labels_by_text.get(key)
        if previous is not None:
            if previous != label:
                raise ValueError(
                    f"Conflicting intent labels in {split_name} for command: {text!r}"
                )
            continue
        labels_by_text[key] = label
        unique.append((text, label))
    return unique


def _validate_fixed_splits(splits):
    seen_in_split = {}
    for split_name, examples in splits.items():
        if not examples:
            raise ValueError(f"The {split_name} intent split is empty")
        normalized = set()
        for text, _ in examples:
            key = _normalize_text(text)
            previous_split = seen_in_split.get(key)
            if previous_split is not None and previous_split != split_name:
                raise ValueError(
                    f"Exact command leakage across {previous_split} and {split_name}: {text!r}"
                )
            seen_in_split[key] = split_name
            normalized.add(key)
    train_labels = {label for _, label in splits["train"]}
    for split_name in ("validation", "test"):
        unseen_labels = {label for _, label in splits[split_name]} - train_labels
        if unseen_labels:
            raise ValueError(
                f"Intents occur in {split_name} but not train: "
                f"{', '.join(sorted(unseen_labels))}"
            )


def _automatic_split(examples, seed):
    grouped = defaultdict(list)
    for example in examples:
        grouped[example[1]].append(example)

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for intent, intent_examples in sorted(grouped.items()):
        if len(intent_examples) < 3:
            raise ValueError(
                f"Intent {intent!r} needs at least 3 examples for an 80/10/10 split"
            )
        rng.shuffle(intent_examples)
        validation_count = max(1, int(len(intent_examples) * 0.10 + 0.5))
        test_count = max(1, int(len(intent_examples) * 0.10 + 0.5))
        while validation_count + test_count >= len(intent_examples):
            if test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                raise ValueError(f"Not enough examples to split intent {intent!r}")

        train_end = len(intent_examples) - validation_count - test_count
        splits["train"].extend(intent_examples[:train_end])
        splits["validation"].extend(intent_examples[train_end:train_end + validation_count])
        splits["test"].extend(intent_examples[train_end + validation_count:])

    for split_name in splits:
        rng.shuffle(splits[split_name])
        splits[split_name] = _deduplicate_split(splits[split_name], split_name)
    _validate_fixed_splits(splits)
    return splits


def load_intent_splits(dataset_dir, seed=42):
    """Use complete fixed splits when present; otherwise split the flat dataset."""
    dataset_dir = Path(dataset_dir)
    fixed_sources = _find_fixed_splits(dataset_dir)

    if fixed_sources:
        splits = {
            split_name: _deduplicate_split(
                _read_intent_source(source), split_name
            )
            for split_name, source in fixed_sources.items()
        }
        _validate_fixed_splits(splits)
        return splits, "fixed"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Intent dataset directory not found: {dataset_dir}")
    examples = _read_intent_source(dataset_dir)
    examples = _deduplicate_split(examples, "full dataset")
    if len({label for _, label in examples}) < 2:
        raise ValueError(f"Expected at least two intent labels in {dataset_dir}")
    return _automatic_split(examples, seed), "automatic 80/10/10"
