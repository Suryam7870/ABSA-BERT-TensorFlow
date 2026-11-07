"""
Train the BERT token-classification model for aspect term extraction.

Usage:
    python -m src.training.train_aspect
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402
from src.models.aspect_model import build_aspect_extraction_model  # noqa: E402
from src.preprocessing.label_alignment import align_labels_with_tokens  # noqa: E402
from src.preprocessing.tokenizer import get_tokenizer  # noqa: E402


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run `python -m src.data.prepare_data` first."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "tokens", "bio_labels"])
    return df


def build_dataset(df: pd.DataFrame, tokenizer) -> dict:
    input_ids, attention_masks, token_type_ids, labels = [], [], [], []

    for row in df.itertuples():
        words = json.loads(row.tokens)
        word_labels = json.loads(row.bio_labels)
        if not words:
            continue
        aligned = align_labels_with_tokens(tokenizer, words, word_labels)
        input_ids.append(aligned["input_ids"])
        attention_masks.append(aligned["attention_mask"])
        token_type_ids.append(aligned["token_type_ids"])
        labels.append(aligned["labels"])

    return {
        "input_ids": np.array(input_ids, dtype=np.int32),
        "attention_mask": np.array(attention_masks, dtype=np.int32),
        "token_type_ids": np.array(token_type_ids, dtype=np.int32),
        "labels": np.array(labels, dtype=np.int32),
    }


def to_tf_dataset(data: dict, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data["input_ids"]), seed=cfg.RANDOM_SEED)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    cfg.ensure_dirs()
    set_seeds(cfg.RANDOM_SEED)

    print("Loading tokenizer ...")
    tokenizer = get_tokenizer()

    print("Loading processed aspect-extraction data ...")
    train_df = load_split(cfg.ASPECT_TRAIN_FILE)
    val_df = load_split(cfg.ASPECT_VAL_FILE)

    if len(train_df) == 0:
        print("No training data available. Aborting.")
        sys.exit(1)

    print(f"Train examples: {len(train_df)} | Val examples: {len(val_df)}")

    print("Tokenizing and aligning labels ...")
    train_data = build_dataset(train_df, tokenizer)
    train_ds = to_tf_dataset(train_data, cfg.BATCH_SIZE, shuffle=True)

    val_ds = None
    if len(val_df) > 0:
        val_data = build_dataset(val_df, tokenizer)
        val_ds = to_tf_dataset(val_data, cfg.EVAL_BATCH_SIZE, shuffle=False)

    print("Building BERT token-classification model ...")
    model = build_aspect_extraction_model()

    os.makedirs(cfg.ASPECT_MODEL_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(cfg.ASPECT_MODEL_DIR, "checkpoint")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds is not None else "loss",
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    epochs = 1 if cfg.DEBUG_MODE else cfg.NUM_EPOCHS

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Saving model + tokenizer to {cfg.ASPECT_MODEL_DIR}")
    model.save_pretrained(cfg.ASPECT_MODEL_DIR)
    tokenizer.save_pretrained(cfg.ASPECT_MODEL_DIR)

    history_path = os.path.join(cfg.REPORTS_DIR, "aspect_model_history.json")
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
