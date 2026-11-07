"""
Bi-LSTM baseline model for aspect-based sentiment classification.

This is intentionally much simpler than the BERT model and serves as a
point of comparison ("Bi-LSTM vs BERT") in the evaluation report. It takes
`[aspect] [SEP] [review text]` as a single whitespace-tokenized sequence and
predicts one of {negative, neutral, positive}.

Architecture:
    Embedding -> Bidirectional LSTM -> Dropout -> Dense -> Softmax
"""

import json
import os
import re
from collections import Counter
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import cfg

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEP_TOKEN = "<SEP>"


def simple_tokenize(text: str) -> List[str]:
    """Lightweight whitespace + punctuation tokenizer (no external deps)."""
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens


class Vocabulary:
    """A minimal word<->id vocabulary with save/load support."""

    def __init__(self, max_size: int = 20000):
        self.max_size = max_size
        self.token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1, SEP_TOKEN: 2}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def build(self, texts: List[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))
        most_common = counter.most_common(self.max_size - len(self.token_to_id))
        for token, _ in most_common:
            if token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token

    def encode(self, tokens: List[str], max_length: int) -> List[int]:
        ids = [self.token_to_id.get(t, self.token_to_id[UNK_TOKEN]) for t in tokens]
        ids = ids[:max_length]
        ids += [self.token_to_id[PAD_TOKEN]] * (max_length - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.token_to_id = json.load(f)
        vocab.id_to_token = {v: k for k, v in vocab.token_to_id.items()}
        return vocab


def encode_aspect_sentiment_example(vocab: Vocabulary, text: str, aspect: str, max_length: int) -> List[int]:
    """Encode `aspect <SEP> review text` as a padded id sequence."""
    tokens = simple_tokenize(aspect) + [SEP_TOKEN] + simple_tokenize(text)
    return vocab.encode(tokens, max_length)


def build_bilstm_model(vocab_size: int, num_classes: int = None) -> keras.Model:
    """
    Build and compile the Bi-LSTM baseline model.

    Embedding -> Bidirectional LSTM -> Dropout -> Dense -> Output(softmax)
    """
    num_classes = num_classes or cfg.NUM_SENTIMENT_LABELS

    model = keras.Sequential(
        [
            layers.Input(shape=(cfg.MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_ids"),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=cfg.EMBEDDING_DIM,
                mask_zero=True,
                name="embedding",
            ),
            layers.Bidirectional(
                layers.LSTM(cfg.LSTM_UNITS, return_sequences=False), name="bilstm"
            ),
            layers.Dropout(cfg.DROPOUT_RATE, name="dropout"),
            layers.Dense(64, activation="relu", name="dense_hidden"),
            layers.Dropout(cfg.DROPOUT_RATE, name="dropout_2"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="bilstm_absa_baseline",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.BILSTM_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
