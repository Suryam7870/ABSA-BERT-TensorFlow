"""
Train the BERT sequence-classification model for aspect sentiment
classification. Input format: [CLS] review text [SEP] aspect term [SEP]

Usage:
    python -m src.training.train_sentiment
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402
from src.models.sentiment_model import build_sentiment_model  # noqa: E402
from src.preprocessing.tokenizer import encode_sentence_pair, get_tokenizer  # noqa: E402


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run `python -m src.data.prepare_data` first."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "aspect", "sentiment"])
    df = df[df["sentiment"].isin(cfg.SENTIMENT_TO_ID.keys())]
    return df


def build_arrays(df: pd.DataFrame):
    input_ids, attention_masks, token_type_ids = [], [], []
    for row in df.itertuples():
        encoded = encode_sentence_pair(str(row.text), str(row.aspect))
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        token_type_ids.append(encoded["token_type_ids"])

    labels = df["sentiment"].map(cfg.SENTIMENT_TO_ID).values.astype(np.int32)

    features = {
        "input_ids": np.array(input_ids, dtype=np.int32),
        "attention_mask": np.array(attention_masks, dtype=np.int32),
        "token_type_ids": np.array(token_type_ids, dtype=np.int32),
    }
    return features, labels


def to_tf_dataset(features: dict, labels: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=cfg.RANDOM_SEED)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    cfg.ensure_dirs()
    set_seeds(cfg.RANDOM_SEED)

    print("Loading tokenizer ...")
    tokenizer = get_tokenizer()

    print("Loading processed sentiment data ...")
    train_df = load_split(cfg.SENTIMENT_TRAIN_FILE)
    val_df = load_split(cfg.SENTIMENT_VAL_FILE)

    if len(train_df) == 0:
        print("No training data available. Aborting.")
        sys.exit(1)

    print(f"Train examples: {len(train_df)} | Val examples: {len(val_df)}")

    print("Encoding examples ...")
    train_features, train_labels = build_arrays(train_df)
    train_ds = to_tf_dataset(train_features, train_labels, cfg.BATCH_SIZE, shuffle=True)

    val_ds = None
    if len(val_df) > 0:
        val_features, val_labels = build_arrays(val_df)
        val_ds = to_tf_dataset(val_features, val_labels, cfg.EVAL_BATCH_SIZE, shuffle=False)

    print("Building BERT sequence-classification model ...")
    model = build_sentiment_model()

    os.makedirs(cfg.SENTIMENT_MODEL_DIR, exist_ok=True)

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

    print(f"Saving model + tokenizer to {cfg.SENTIMENT_MODEL_DIR}")
    model.save_pretrained(cfg.SENTIMENT_MODEL_DIR)
    tokenizer.save_pretrained(cfg.SENTIMENT_MODEL_DIR)

    history_path = os.path.join(cfg.REPORTS_DIR, "sentiment_model_history.json")
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
