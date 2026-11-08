import argparse
import json
import pickle
import re
import statistics
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
NLP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_TEST_FILE = (
    NLP_DIR / "data" / "entities" / "evaluation v2" / "winpilot_unseen_test_300.jsonl"
)
DEFAULT_INTENT_MODEL = NLP_DIR / "models" / "intent" / "base_intent_model_v2(w+char).pkl"
DEFAULT_NER_MODEL = NLP_DIR / "models" / "ner" / "distilbert"
DEFAULT_RESULTS_FILE = SCRIPT_DIR / "evaluation_results.jsonl"
MAX_LENGTH = 256


def load_jsonl(path):
    """Read and validate unseen command and entity records.

    Args:
        path - JSONL dataset path.

    Returns:
        Validated evaluation records.

    Raises:
        FileNotFoundError - If the dataset file does not exist.
        ValueError - If a record or entity span is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {path}\n"
            "Place winpilot_unseen_test_300.jsonl in "
            f"{path.parent} or pass --dataset PATH."
        )

    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error

            required = {"text", "intent", "entities"}
            missing = required - item.keys()
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: "
                    f"{', '.join(sorted(missing))}"
                )
            if not isinstance(item["text"], str):
                raise ValueError(f"Line {line_number}: 'text' must be a string")
            if not isinstance(item["intent"], str):
                raise ValueError(f"Line {line_number}: 'intent' must be a string")
            if not isinstance(item["entities"], list):
                raise ValueError(f"Line {line_number}: 'entities' must be a list")

            for entity in item["entities"]:
                if not isinstance(entity, dict) or not {"start", "end", "label"} <= entity.keys():
                    raise ValueError(
                        f"Line {line_number}: each entity needs start, end, and label"
                    )
                start, end, label = entity["start"], entity["end"], entity["label"]
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError(
                        f"Line {line_number}: entity offsets must be character indexes"
                    )
                if not isinstance(label, str) or not (0 <= start < end <= len(item["text"])):
                    raise ValueError(
                        f"Line {line_number}: invalid entity span or label {entity!r}"
                    )

            records.append(item)

    if not records:
        raise ValueError(f"No test records found in {path}")
    return records


class IntentPredictor:
    """Adapter for the project's saved scikit-learn intent pipeline."""

    def __init__(self, model_path):
        """Load the saved intent classification pipeline.

        Args:
            model_path - Path to the serialized model.

        Returns:
            None.

        Raises:
            FileNotFoundError - If the model file does not exist.
        """
        if not model_path.is_file():
            raise FileNotFoundError(f"Intent model not found: {model_path}")
        with model_path.open("rb") as stream:
            self.model = pickle.load(stream)

    def predict(self, text):
        """Predict the intent label for one command.

        Args:
            text - Command text to classify.

        Returns:
            Predicted intent label.
        """
        return str(self.model.predict([text])[0])


class DistilBertEntityPredictor:
    """Predict BIO entities and map them back to original character spans."""

    def __init__(self, model_dir, device):
        """Load the tokenizer and entity model on the requested device.

        Args:
            model_dir - Directory containing the saved model and tokenizer.
            device - Torch device used for model inference.

        Returns:
            None.

        Raises:
            FileNotFoundError - If the model directory does not exist.
        """
        if not model_dir.is_dir():
            raise FileNotFoundError(f"DistilBERT NER model not found: {model_dir}")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        self.id_to_tag = {
            int(index): tag
            for index, tag in self.model.config.id2label.items()
        }
        self.max_length = min(
            MAX_LENGTH,
            self.model.config.max_position_embeddings,
        )

    def predict(self, text):
        """Predict entity labels and return their character spans.

        Args:
            text - Command text to analyse.

        Returns:
            Entity dictionaries with start, end, and label fields.
        """
        word_matches = list(re.finditer(r"\S+", text))
        if not word_matches:
            return []

        words = [match.group(0) for match in word_matches]
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }

        with torch.inference_mode():
            predictions = (
                self.model(**model_inputs)
                .logits.argmax(dim=-1)[0]
                .cpu()
                .tolist()
            )

        word_tags = []
        previous_word_id = None
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue
            word_tags.append((word_id, self.id_to_tag[predictions[position]]))
            previous_word_id = word_id

        entities = []
        active_label = None
        active_start = None
        active_end = None

        def close_entity():
            """Append the active entity span when one is open.

            Returns:
                None.
            """
            if active_label is not None:
                # Whitespace tokenization includes sentence punctuation in the
                # final word (for example, ``report.docx.``). Exclude common
                # sentence-ending punctuation from the predicted span so it
                # follows the character boundaries used by the gold data.
                entity_end = active_end
                while (
                    entity_end > active_start
                    and text[entity_end - 1] in ".,!?;:"
                ):
                    entity_end -= 1
                entities.append({
                    "start": active_start,
                    "end": entity_end,
                    "label": active_label,
                })

        for word_id, tag in word_tags:
            if tag.startswith("B-"):
                close_entity()
                active_label = tag[2:]
                active_start = word_matches[word_id].start()
                active_end = word_matches[word_id].end()
            elif tag.startswith("I-"):
                label = tag[2:]
                if active_label != label:
                    close_entity()
                    active_label = label
                    active_start = word_matches[word_id].start()
                active_end = word_matches[word_id].end()
            else:
                close_entity()
                active_label = None
                active_start = None
                active_end = None

        close_entity()
        return entities


def exact_entity_counts(gold, predicted):
    """Count exact entity matches, false positives, and false negatives.

    Args:
        gold - Expected entity span dictionaries.
        predicted - Predicted entity span dictionaries.

    Returns:
        True positive, false positive, and false negative counts.
    """
    gold_set = {
        (entity["start"], entity["end"], entity["label"])
        for entity in gold
    }
    predicted_set = {
        (entity["start"], entity["end"], entity["label"])
        for entity in predicted
    }
    return (
        len(gold_set & predicted_set),
        len(predicted_set - gold_set),
        len(gold_set - predicted_set),
    )


def synchronize_if_cuda(device):
    """Wait for pending work when evaluation runs on a CUDA device.

    Args:
        device - Device used for model inference.

    Returns:
        None.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    """Evaluate intent and entity models on an unseen command dataset.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate intent and DistilBERT NER on unseen commands."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--intent-model", type=Path, default=DEFAULT_INTENT_MODEL)
    parser.add_argument("--ner-model", type=Path, default=DEFAULT_NER_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device for DistilBERT (default: CUDA when available).",
    )
    args = parser.parse_args()

    data = load_jsonl(args.dataset)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )

    intent_model = IntentPredictor(args.intent_model)
    entity_model = DistilBertEntityPredictor(args.ner_model, device)

    # Warm up both models so first-call initialization does not distort latency.
    intent_model.predict(data[0]["text"])
    entity_model.predict(data[0]["text"])

    intent_gold, intent_predicted = [], []
    entity_tp = entity_fp = entity_fn = 0
    latencies_ms = []
    results = []

    for index, item in enumerate(data, start=1):
        text = item["text"]

        synchronize_if_cuda(device)
        start_time = time.perf_counter()
        predicted_intent = intent_model.predict(text)
        predicted_entities = entity_model.predict(text)
        synchronize_if_cuda(device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies_ms.append(elapsed_ms)

        gold_intent = item["intent"]
        gold_entities = item["entities"]
        true_positive, false_positive, false_negative = exact_entity_counts(
            gold_entities,
            predicted_entities,
        )
        entity_tp += true_positive
        entity_fp += false_positive
        entity_fn += false_negative
        intent_gold.append(gold_intent)
        intent_predicted.append(predicted_intent)

        gold_set = {
            (entity["start"], entity["end"], entity["label"])
            for entity in gold_entities
        }
        predicted_set = {
            (entity["start"], entity["end"], entity["label"])
            for entity in predicted_entities
        }
        results.append({
            "id": item.get("id", index),
            "text": text,
            "expected_intent": gold_intent,
            "predicted_intent": predicted_intent,
            "expected_entities": gold_entities,
            "predicted_entities": predicted_entities,
            "intent_correct": predicted_intent == gold_intent,
            "entities_exact": gold_set == predicted_set,
            "latency_ms": round(elapsed_ms, 3),
        })

    precision = entity_tp / (entity_tp + entity_fp) if entity_tp + entity_fp else 0.0
    recall = entity_tp / (entity_tp + entity_fn) if entity_tp + entity_fn else 0.0
    entity_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    sorted_latencies = sorted(latencies_ms)
    p95_index = int(0.95 * (len(sorted_latencies) - 1))

    print_section = "=" * 60
    print(print_section)
    print("UNSEEN COMMAND EVALUATION")
    print(print_section)
    print(f"Dataset             : {args.dataset}")
    print(f"Intent model        : {args.intent_model}")
    print(f"NER model           : {args.ner_model}")
    print(f"NER device          : {device}")
    print(f"Commands            : {len(data)}")
    print(f"Intent accuracy     : {accuracy_score(intent_gold, intent_predicted):.4f}")
    print(f"Exact entity precision: {precision:.4f}")
    print(f"Exact entity recall : {recall:.4f}")
    print(f"Exact entity F1     : {entity_f1:.4f}")
    print(f"Median latency      : {statistics.median(latencies_ms):.2f} ms")
    print(f"P95 latency         : {sorted_latencies[p95_index]:.2f} ms")

    print("\nIntent report:")
    print(classification_report(intent_gold, intent_predicted, zero_division=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Per-command results : {args.output}")


if __name__ == "__main__":
    main()
