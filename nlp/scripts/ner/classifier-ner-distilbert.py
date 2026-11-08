"""Run the fine-tuned local DistilBERT NER model on one command."""

from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "ner" / "distilbert"


class DistilBertNER:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_dir}. "
                "Run nlp/train/train_ner_distilbert.py first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id_to_tag = {int(key): value for key, value in self.model.config.id2label.items()}

    def predict(self, text):
        words = text.split()
        if not words:
            return [], []

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = encoded.word_ids(batch_index=0)
        with torch.inference_mode():
            predictions = self.model(**encoded).logits.argmax(dim=-1)[0].tolist()

        output_words, tags = [], []
        previous_word_id = None
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue
            output_words.append(words[word_id])
            tags.append(self.id_to_tag[predictions[position]])
            previous_word_id = word_id
        return output_words, tags


def extract_entities(words, tags):
    entities = []
    active_type = None
    current_words = []

    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_words:
                entities.append((active_type, " ".join(current_words)))
            active_type = tag[2:]
            current_words = [word]
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if active_type == entity_type:
                current_words.append(word)
            else:
                if current_words:
                    entities.append((active_type, " ".join(current_words)))
                active_type = entity_type
                current_words = [word]
        else:
            if current_words:
                entities.append((active_type, " ".join(current_words)))
            active_type = None
            current_words = []

    if current_words:
        entities.append((active_type, " ".join(current_words)))
    return entities


def main():
    ner = DistilBertNER()
    print("WinPilot DistilBERT NER")
    print("Type 'exit' to quit.")
    while True:
        text = input("Command: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue
        words, tags = ner.predict(text)
        print("\n".join(f"{word:<30} {tag}" for word, tag in zip(words, tags)))
        print("Entities:")
        for entity_type, value in extract_entities(words, tags):
            print(f"{entity_type}: {value}")
        print()


if __name__ == "__main__":
    main()
