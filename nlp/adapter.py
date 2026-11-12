from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from core.command import Command

logger = logging.getLogger(__name__)

NLP_DIR = Path(__file__).resolve().parent
INTENT_MODEL_PATH = NLP_DIR / "models" / "intent" / "base_intent_model_v2(w+char).pkl"
NER_MODEL_DIR = NLP_DIR / "models" / "ner" / "distilbert"


class LocalNLPInterpreter:
    """Interpret parser misses with confidence-gated local models.

    Models are loaded only when a command misses the deterministic parser.
    The intent threshold is intentionally conservative and can be tuned with
    held-out command data; entity confidence is checked separately.
    """

    def __init__(
        self,
        intent_threshold: float = 0.80,
        entity_threshold: float = 0.70,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        """Initialise the interpreter and confidence thresholds.

        Args:
            intent_threshold - Minimum confidence required for an intent.
            entity_threshold - Minimum token confidence required for an entity.

        Returns:
            None.
        """
        self.intent_threshold = intent_threshold
        self.entity_threshold = entity_threshold
        self.reporter = reporter
        self._intent_model: Any | None = None
        self._ner_model: Any | None = None
        self._load_error: str | None = None

    def _report(self, message: str) -> None:
        """Write a local-model diagnostic to logs and, when configured, the CLI."""
        logger.info(message)
        if self.reporter is not None:
            self.reporter(message)

    @property
    def models_loaded(self) -> bool:
        """Whether both local intent and entity models are ready."""
        return self._intent_model is not None and self._ner_model is not None

    def _load_models(self) -> bool:
        """Load the intent and entity models once on first use.

        Returns:
            True if both models are available, otherwise False.
        """
        if self._intent_model is not None and self._ner_model is not None:
            return True
        if self._load_error is not None:
            return False

        started = time.perf_counter()
        logger.info("Loading local intent and entity models")
        try:
            import pickle

            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            with INTENT_MODEL_PATH.open("rb") as stream:
                self._intent_model = pickle.load(stream)

            tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_DIR, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_DIR)
            model.eval()
            self._ner_model = (torch, tokenizer, model)
            logger.info(
                "Local NLP models loaded in %.1f seconds",
                time.perf_counter() - started,
            )
            return True
        except Exception as error:
            self._load_error = str(error)
            logger.warning(
                "Local NLP models are unavailable after %.1f seconds: %s",
                time.perf_counter() - started,
                error,
            )
            if self.reporter is not None:
                self.reporter(f"Local NLP unavailable: {error}")
            self._intent_model = None
            self._ner_model = None
            return False

    def _entities(self, text: str) -> list[tuple[str, str, float]]:
        """Extract entity spans and their lowest token confidence.

        Args:
            text - Command text to analyse.

        Returns:
            Entity label, text, and confidence tuples.
        """
        if not self._load_models():
            return []

        torch, tokenizer, model = self._ner_model
        words = text.split()
        if not words:
            return []

        encoded = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = encoded.word_ids(batch_index=0)
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        id_to_tag = {
            int(key): value
            for key, value in model.config.id2label.items()
        }
        word_predictions: list[tuple[str, str, float]] = []
        previous_word_id: int | None = None
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue
            scores = probabilities[position]
            label_id = int(scores.argmax().item())
            word_predictions.append(
                (words[word_id], id_to_tag[label_id], float(scores[label_id].item()))
            )
            previous_word_id = word_id

        entities: list[tuple[str, str, float]] = []
        active_label: str | None = None
        active_words: list[str] = []
        active_scores: list[float] = []

        def flush() -> None:
            """Append the active entity to the result when one is present.

            Returns:
                None.
            """
            if active_label and active_words:
                entities.append(
                    (active_label, " ".join(active_words), min(active_scores))
                )

        for word, tag, score in word_predictions:
            if tag.startswith("B-"):
                flush()
                active_label = tag[2:]
                active_words = [word]
                active_scores = [score]
            elif tag.startswith("I-") and active_label == tag[2:]:
                active_words.append(word)
                active_scores.append(score)
            else:
                flush()
                active_label = None
                active_words = []
                active_scores = []
        flush()
        if entities:
            predictions = "; ".join(
                f"{label}={value!r} ({score:.3f})"
                for label, value, score in entities
            )
            self._report(f"Local NLP entity predictions: {predictions}")
        else:
            self._report("Local NLP entity predictions: none")
        return entities

    def interpret(self, text: str) -> Command | None:
        """Return a command only when intent and required entities are clear.

        Args:
            text - User command to interpret.

        Returns:
            Parsed command, or None when confidence or required entities are low.
        """
        if not self._load_models():
            return None

        model = self._intent_model
        probabilities = model.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        intent = str(model.classes_[best_index])
        confidence = float(probabilities[best_index])
        self._report(
            f"Local NLP intent prediction: {intent} (confidence {confidence:.3f})"
        )
        if confidence < self.intent_threshold:
            self._report(
                f"Local NLP rejected intent: confidence {confidence:.3f} is below "
                f"the {self.intent_threshold:.2f} threshold"
            )
            return None

        move = re.search(r"\s+to\s+", text, flags=re.IGNORECASE)
        if intent.startswith(("move_", "copy_")):
            if move is None:
                return None
            source_text = text[:move.start()]
            destination_text = text[move.end():]
            source_entities = self._entities(source_text)
            destination_entities = self._entities(destination_text)
            source_entity = self._confident_entity(source_entities)
            destination_entity = self._confident_entity(destination_entities)
            if source_entity is None or destination_entity is None:
                self._report(
                    "Local NLP rejected command: could not extract confident "
                    "source and destination entities"
                )
                return None
            self._report("Local NLP accepted command with confident source and destination")
            return Command(
                "move" if intent.startswith("move_") else "copy",
                {"source": source_entity[1], "destination": destination_entity[1]},
            )

        entities = self._entities(text)
        allowed = {
            "open_application": {"APP"},
            "create_file": {"FILE", "PATH_FILE"},
            "create_folder": {"FOLDER", "PATH_DIR"},
            "delete_file": {"FILE", "PATH_FILE"},
            "delete_folder": {"FOLDER", "PATH_DIR"},
            "rename_file": {"FILE", "PATH_FILE"},
            "rename_folder": {"FOLDER", "PATH_DIR"},
        }.get(intent)
        if not allowed:
            self._report(f"Local NLP rejected command: unsupported intent {intent}")
            return None

        entity = next(
            (
                (label, value, score)
                for label, value, score in entities
                if label in allowed and score >= self.entity_threshold
            ),
            None,
        )
        if entity is None:
            self._report(
                f"Local NLP rejected command: expected entity {sorted(allowed)} "
                f"with confidence at least {self.entity_threshold:.2f}"
            )
            return None

        _, target, _ = entity
        if intent.startswith("rename_"):
            if move is None:
                return None
            new_name = text[move.end():].strip()
            if not new_name:
                self._report("Local NLP rejected rename: new name was missing")
                return None
            self._report("Local NLP accepted rename command")
            return Command(intent, {"target": target, "new_name": new_name})

        requires_approval = intent.startswith("delete_")
        self._report(f"Local NLP accepted command: {intent}")
        return Command(intent, {"target": target}, requires_approval=requires_approval)

    def _confident_entity(
        self,
        entities: list[tuple[str, str, float]],
    ) -> tuple[str, str, float] | None:
        """Select the first entity that meets the configured threshold.

        Args:
            entities - Entity label, text, and confidence tuples.

        Returns:
            The first sufficiently confident entity, or None.
        """
        return next(
            (entity for entity in entities if entity[2] >= self.entity_threshold),
            None,
        )
