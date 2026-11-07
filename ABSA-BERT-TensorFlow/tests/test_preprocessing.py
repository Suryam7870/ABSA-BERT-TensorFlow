"""
Unit tests for tokenization, subword-label alignment, and data preparation
utilities. These tests do not require any trained model — they only exercise
the pretrained BERT tokenizer (downloaded automatically by `transformers`
the first time it is used) and pure-Python data logic.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg  # noqa: E402
from src.data.prepare_data import build_bio_labels, clean_records  # noqa: E402
from src.preprocessing.label_alignment import (  # noqa: E402
    align_labels_with_tokens,
    decode_predictions_to_words,
)
from src.preprocessing.tokenizer import encode_sentence_pair, get_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


def test_tokenizer_loads(tokenizer):
    assert tokenizer is not None
    assert tokenizer.cls_token is not None
    assert tokenizer.sep_token is not None
    assert tokenizer.pad_token is not None


def test_encode_sentence_pair_shapes():
    encoded = encode_sentence_pair("The food was great.", "food", max_length=32)
    assert len(encoded["input_ids"]) == 32
    assert len(encoded["attention_mask"]) == 32
    assert len(encoded["token_type_ids"]) == 32
    # attention_mask should contain at least one non-pad token
    assert sum(encoded["attention_mask"]) > 0


def test_build_bio_labels_simple():
    text = "The battery life is excellent"
    aspects = [{"term": "battery life", "polarity": "positive"}]
    tokens, labels = build_bio_labels(text, aspects)
    assert tokens == ["The", "battery", "life", "is", "excellent"]
    assert labels == ["O", "B-ASPECT", "I-ASPECT", "O", "O"]


def test_build_bio_labels_multiple_aspects():
    text = "The food was amazing but the service was slow"
    aspects = [
        {"term": "food", "polarity": "positive"},
        {"term": "service", "polarity": "negative"},
    ]
    tokens, labels = build_bio_labels(text, aspects)
    assert labels[tokens.index("food")] == "B-ASPECT"
    assert labels[tokens.index("service")] == "B-ASPECT"


def test_build_bio_labels_missing_aspect_is_skipped():
    text = "Great value for money"
    aspects = [{"term": "not present here", "polarity": "positive"}]
    tokens, labels = build_bio_labels(text, aspects)
    assert all(label == "O" for label in labels)


def test_clean_records_drops_missing_and_invalid():
    records = [
        {"text": "Good food", "aspects": [{"term": "food", "polarity": "positive"}]},
        {"text": "", "aspects": []},  # missing text -> dropped
        {"text": None, "aspects": []},  # missing text -> dropped
        {"text": "Bad service", "aspects": [{"term": "service", "polarity": "conflict"}]},  # invalid polarity
        {"text": "Good food", "aspects": [{"term": "food", "polarity": "positive"}]},  # duplicate
    ]
    cleaned = clean_records(records)
    assert len(cleaned) == 2  # "Good food" (kept once) + "Bad service" (aspects emptied)
    texts = [r["text"] for r in cleaned]
    assert "Good food" in texts
    assert "Bad service" in texts
    bad_service_record = next(r for r in cleaned if r["text"] == "Bad service")
    assert bad_service_record["aspects"] == []


def test_label_alignment_lengths_match(tokenizer):
    words = "The battery life is excellent".split()
    word_labels = ["O", "B-ASPECT", "I-ASPECT", "O", "O"]
    aligned = align_labels_with_tokens(tokenizer, words, word_labels, max_length=16)
    assert len(aligned["input_ids"]) == 16
    assert len(aligned["labels"]) == 16


def test_label_alignment_ignores_special_tokens(tokenizer):
    words = "battery life".split()
    word_labels = ["B-ASPECT", "I-ASPECT"]
    aligned = align_labels_with_tokens(tokenizer, words, word_labels, max_length=16)
    # First token should be [CLS] -> ignored
    assert aligned["labels"][0] == cfg.IGNORE_LABEL_ID
    # Somewhere in the sequence, padding should also be ignored.
    assert cfg.IGNORE_LABEL_ID in aligned["labels"]


def test_label_alignment_raises_on_length_mismatch(tokenizer):
    with pytest.raises(ValueError):
        align_labels_with_tokens(tokenizer, ["one", "two"], ["O"], max_length=16)


def test_decode_predictions_to_words_roundtrip(tokenizer):
    words = "battery life is excellent".split()
    encoded = tokenizer(words, is_split_into_words=True, max_length=16,
                         truncation=True, padding="max_length")
    word_ids = encoded.word_ids()
    # Fake predicted label ids: 1=B-ASPECT for first token position of each word, else 0=O
    predicted_ids = [0] * len(word_ids)
    seen = set()
    for i, wid in enumerate(word_ids):
        if wid is not None and wid not in seen:
            predicted_ids[i] = 1 if wid == 0 else 0
            seen.add(wid)
    decoded = decode_predictions_to_words(word_ids, predicted_ids)
    assert len(decoded) == len(words)
    assert decoded[0] == "B-ASPECT"
