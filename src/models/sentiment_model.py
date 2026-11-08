"""
BERT-based sequence classification model for aspect sentiment classification.

Input format: `[CLS] review text [SEP] aspect term [SEP]` — this lets the
[CLS] representation attend to both the full review context and the specific
aspect being classified, which is important for sentences with multiple
aspects of different polarity (e.g. "food great, service terrible").

Uses `TFAutoModelForSequenceClassification` (TensorFlow / Keras — no PyTorch).
"""

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

from config import cfg


def build_sentiment_model(model_name: str = None, num_labels: int = None):
    """
    Build a TensorFlow BERT sequence-classification model for 3-class
    (negative / neutral / positive) aspect sentiment classification.
    """
    name = model_name or cfg.BERT_MODEL_NAME
    n_labels = num_labels or cfg.NUM_SENTIMENT_LABELS

    model = TFAutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=n_labels,
        id2label=cfg.ID_TO_SENTIMENT,
        label2id=cfg.SENTIMENT_TO_ID,
    )

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=cfg.LEARNING_RATE,
        decay_steps=1000,
        end_learning_rate=0.0,
        power=1.0,
    )
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
