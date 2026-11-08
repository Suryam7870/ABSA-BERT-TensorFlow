"""
BERT-based token classification model for aspect term extraction (BIO tagging).

Uses `TFAutoModelForTokenClassification` from Hugging Face Transformers
(TensorFlow / Keras backend only — no PyTorch).
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from transformers import TFAutoModelForTokenClassification

from config import cfg


def build_aspect_extraction_model(model_name: str = None, num_labels: int = None):
    """
    Build a TensorFlow BERT token-classification model for BIO aspect tagging.

    Args:
        model_name: Pretrained checkpoint name (defaults to config.BERT_MODEL_NAME).
        num_labels: Number of BIO labels (defaults to config.NUM_BIO_LABELS).

    Returns:
        A compiled `TFAutoModelForTokenClassification` instance.
    """
    name = model_name or cfg.BERT_MODEL_NAME
    n_labels = num_labels or cfg.NUM_BIO_LABELS

    model = TFAutoModelForTokenClassification.from_pretrained(
    name,
    num_labels=len(cfg.LABEL_TO_ID),
    id2label={v: k for k, v in cfg.LABEL_TO_ID.items()},
    label2id=cfg.LABEL_TO_ID,
    use_safetensors=False,
    )

    optimizer = _build_optimizer()
    # The HF TF models compute their own loss internally when `labels` is
    # passed as part of the input dict, but we compile with `loss=None` and
    # rely on `model(..., labels=...)` during training via `train_step`
    # override below is unnecessary — HF TF models support `.fit()` directly
    # when compiled with loss=None because they add the loss internally.
    model.compile(optimizer=optimizer)
    return model


def _build_optimizer():
    """Adam optimizer with a linearly decaying learning rate (AdamW-style decay)."""
    import tensorflow as tf

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=cfg.LEARNING_RATE,
        decay_steps=1000,
        end_learning_rate=0.0,
        power=1.0,
    )
    try:
        # AdamW is available in tf.keras.optimizers from TF 2.11+
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer
