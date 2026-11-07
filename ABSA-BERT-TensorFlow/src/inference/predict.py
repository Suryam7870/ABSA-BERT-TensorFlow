"""
End-to-end Aspect-Based Sentiment Analysis inference pipeline.

Combines:
  Stage A: BERT token classification -> aspect term extraction (BIO tagging)
  Stage B: BERT sequence classification -> per-aspect sentiment

Usage:
    from src.inference.predict import predict_absa
    result = predict_absa("The food was amazing but the service was very slow.")

Returns a list of dicts:
    [
        {"aspect": "food", "sentiment": "positive", "confidence": 0.96},
        {"aspect": "service", "sentiment": "negative", "confidence": 0.94},
    ]

Confidence is the softmax probability of the predicted sentiment class from
the sentiment model. It reflects the model's own certainty and has NOT been
calibrated (e.g. via temperature scaling), so it should be treated as a
relative signal rather than a precise probability.
"""

import os
import sys
from functools import lru_cache
from typing import List, TypedDict

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402
from src.preprocessing.label_alignment import decode_predictions_to_words  # noqa: E402
from src.preprocessing.tokenizer import encode_sentence_pair, get_tokenizer  # noqa: E402


class AspectSentiment(TypedDict):
    aspect: str
    sentiment: str
    confidence: float


class ModelsNotTrainedError(RuntimeError):
    """Raised when predict_absa() is called before models have been trained."""


@lru_cache(maxsize=1)
def _load_aspect_model():
    from transformers import TFAutoModelForTokenClassification

    if not os.path.isdir(cfg.ASPECT_MODEL_DIR) or not os.listdir(cfg.ASPECT_MODEL_DIR):
        raise ModelsNotTrainedError(
            f"No trained aspect-extraction model found at '{cfg.ASPECT_MODEL_DIR}'. "
            f"Run `python -m src.training.train_aspect` first."
        )
    return TFAutoModelForTokenClassification.from_pretrained(cfg.ASPECT_MODEL_DIR)


@lru_cache(maxsize=1)
def _load_sentiment_model():
    from transformers import TFAutoModelForSequenceClassification

    if not os.path.isdir(cfg.SENTIMENT_MODEL_DIR) or not os.listdir(cfg.SENTIMENT_MODEL_DIR):
        raise ModelsNotTrainedError(
            f"No trained sentiment model found at '{cfg.SENTIMENT_MODEL_DIR}'. "
            f"Run `python -m src.training.train_sentiment` first."
        )
    return TFAutoModelForSequenceClassification.from_pretrained(cfg.SENTIMENT_MODEL_DIR)


def clear_model_cache() -> None:
    """Clear cached models — useful for tests or after retraining."""
    _load_aspect_model.cache_clear()
    _load_sentiment_model.cache_clear()


def extract_aspects(text: str) -> List[str]:
    """
    Run the aspect-extraction (token classification) model over `text` and
    return the list of extracted aspect term strings (surface forms as they
    appear in the original text, in order of first appearance).
    """
    tokenizer = get_tokenizer()
    model = _load_aspect_model()

    words = text.split()
    if not words:
        return []

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        max_length=cfg.MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
        return_token_type_ids=True,
    )
    word_ids = encoded.word_ids()

    logits = model(
        input_ids=tf.constant([encoded["input_ids"]]),
        attention_mask=tf.constant([encoded["attention_mask"]]),
        token_type_ids=tf.constant([encoded["token_type_ids"]]),
    ).logits
    pred_ids = tf.argmax(logits, axis=-1).numpy()[0]

    word_labels = decode_predictions_to_words(word_ids, pred_ids)

    # Group consecutive B-ASPECT / I-ASPECT word labels into aspect spans.
    aspects = []
    current_span = []
    for word, label in zip(words, word_labels):
        if label == "B-ASPECT":
            if current_span:
                aspects.append(" ".join(current_span))
            current_span = [word]
        elif label == "I-ASPECT" and current_span:
            current_span.append(word)
        else:
            if current_span:
                aspects.append(" ".join(current_span))
                current_span = []
    if current_span:
        aspects.append(" ".join(current_span))

    # De-duplicate while preserving order.
    seen = set()
    unique_aspects = []
    for aspect in aspects:
        key = aspect.lower().strip(".,!?;:")
        if key and key not in seen:
            seen.add(key)
            unique_aspects.append(aspect.strip(".,!?;:"))
    return unique_aspects


def classify_sentiment(text: str, aspect: str) -> AspectSentiment:
    """
    Classify the sentiment of a single aspect within its sentence context
    using the BERT sequence-classification model.
    """
    model = _load_sentiment_model()
    encoded = encode_sentence_pair(text, aspect)

    logits = model(
        input_ids=tf.constant([encoded["input_ids"]]),
        attention_mask=tf.constant([encoded["attention_mask"]]),
        token_type_ids=tf.constant([encoded["token_type_ids"]]),
    ).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred_id = int(np.argmax(probs))

    return {
        "aspect": aspect,
        "sentiment": cfg.ID_TO_SENTIMENT[pred_id],
        "confidence": float(probs[pred_id]),
    }


def predict_absa(text: str) -> List[AspectSentiment]:
    """
    Run the full two-stage ABSA pipeline on a single review.

    Args:
        text: Raw review text, e.g.
              "The food was amazing but the service was very slow."

    Returns:
        A list of {"aspect": str, "sentiment": str, "confidence": float}
        dicts, one per extracted aspect, in order of appearance.

    Raises:
        ModelsNotTrainedError: if the required models have not been trained yet.
        ValueError: if `text` is empty.
    """
    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    text = " ".join(text.split())
    aspects = extract_aspects(text)

    results = [classify_sentiment(text, aspect) for aspect in aspects]
    return results


def overall_sentiment(results: List[AspectSentiment]) -> str:
    """
    Derive a simple overall-sentiment summary label from per-aspect results:
    "positive", "negative", "neutral", "mixed", or "no aspects found".
    """
    if not results:
        return "no aspects found"

    labels = {r["sentiment"] for r in results}
    if labels == {"positive"}:
        return "positive"
    if labels == {"negative"}:
        return "negative"
    if labels == {"neutral"}:
        return "neutral"
    if "positive" in labels and "negative" in labels:
        return "mixed"
    return "neutral"


if __name__ == "__main__":
    example = "The food was amazing but the service was extremely slow."
    print(f"Input: {example}")
    try:
        output = predict_absa(example)
        for item in output:
            print(item)
        print(f"Overall sentiment: {overall_sentiment(output)}")
    except ModelsNotTrainedError as exc:
        print(str(exc))
