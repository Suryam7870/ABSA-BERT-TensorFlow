"""
Train the Bi-LSTM baseline for aspect sentiment classification.

Usage:
    python -m src.training.train_baseline
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402
from src.models.bilstm import (  # noqa: E402
    Vocabulary,
    build_bilstm_model,
    encode_aspect_sentiment_example,
)


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
    return df


def dataframe_to_arrays(df: pd.DataFrame, vocab: Vocabulary):
    X = np.array(
        [
            encode_aspect_sentiment_example(vocab, row.text, row.aspect, cfg.MAX_SEQ_LENGTH)
            for row in df.itertuples()
        ],
        dtype=np.int32,
    )
    y = df["sentiment"].map(cfg.SENTIMENT_TO_ID).values.astype(np.int32)
    return X, y


def main() -> None:
    cfg.ensure_dirs()
    set_seeds(cfg.RANDOM_SEED)

    print("Loading processed sentiment data ...")
    train_df = load_split(cfg.SENTIMENT_TRAIN_FILE)
    val_df = load_split(cfg.SENTIMENT_VAL_FILE)

    if len(train_df) == 0:
        print("No training data available. Aborting.")
        sys.exit(1)

    print(f"Train examples: {len(train_df)} | Val examples: {len(val_df)}")

    print("Building vocabulary ...")
    vocab = Vocabulary(max_size=cfg.VOCAB_SIZE)
    vocab.build(
        [f"{row.aspect} {row.text}" for row in train_df.itertuples()]
    )
    print(f"Vocabulary size: {len(vocab)}")

    X_train, y_train = dataframe_to_arrays(train_df, vocab)
    if len(val_df) > 0:
        X_val, y_val = dataframe_to_arrays(val_df, vocab)
        validation_data = (X_val, y_val)
    else:
        validation_data = None

    print("Building Bi-LSTM model ...")
    model = build_bilstm_model(vocab_size=len(vocab))
    model.summary()

    os.makedirs(cfg.BILSTM_MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(cfg.BILSTM_MODEL_DIR, "bilstm_model.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss" if validation_data else "loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    epochs = 2 if cfg.DEBUG_MODE else cfg.BILSTM_NUM_EPOCHS

    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    vocab.save(cfg.TOKENIZER_VOCAB_PATH)
    print(f"Saved vocabulary to {cfg.TOKENIZER_VOCAB_PATH}")

    history_path = os.path.join(cfg.REPORTS_DIR, "bilstm_history.json")
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f)
    print(f"Saved training history to {history_path}")

    print(f"Best model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
