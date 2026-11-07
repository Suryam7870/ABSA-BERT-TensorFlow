"""
Central configuration for the Aspect-Based Sentiment Analysis (ABSA) project.

All paths, hyperparameters, and constants used across the project are defined
here so that no script hardcodes values in more than one place.

Import this module wherever configuration values are needed:

    from config import Config
    cfg = Config()
"""

import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    """Single source of truth for project-wide configuration."""

    # -------------------------------------------------------------------
    # General / reproducibility
    # -------------------------------------------------------------------
    RANDOM_SEED: int = 42

    # If True, only a small subset of the data is used and training runs for
    # very few steps. This lets the whole pipeline be exercised quickly on a
    # laptop CPU. Set to False for a full training run.
    DEBUG_MODE: bool = True
    DEBUG_SAMPLE_SIZE: int = 200  # number of examples used when DEBUG_MODE=True

    # -------------------------------------------------------------------
    # Pretrained BERT model
    # -------------------------------------------------------------------
    BERT_MODEL_NAME: str = "bert-base-uncased"
    MAX_SEQ_LENGTH: int = 128

    # -------------------------------------------------------------------
    # BIO labels for aspect extraction (token classification)
    # -------------------------------------------------------------------
    BIO_LABELS: List[str] = field(default_factory=lambda: ["O", "B-ASPECT", "I-ASPECT"])
    LABEL_TO_ID: dict = field(
        default_factory=lambda: {"O": 0, "B-ASPECT": 1, "I-ASPECT": 2}
    )
    ID_TO_LABEL: dict = field(
        default_factory=lambda: {0: "O", 1: "B-ASPECT", 2: "I-ASPECT"}
    )
    IGNORE_LABEL_ID: int = -100  # ignored by the loss function

    # -------------------------------------------------------------------
    # Sentiment labels
    # -------------------------------------------------------------------
    SENTIMENT_LABELS: List[str] = field(
        default_factory=lambda: ["negative", "neutral", "positive"]
    )
    SENTIMENT_TO_ID: dict = field(
        default_factory=lambda: {"negative": 0, "neutral": 1, "positive": 2}
    )
    ID_TO_SENTIMENT: dict = field(
        default_factory=lambda: {0: "negative", 1: "neutral", 2: "positive"}
    )
    NUM_SENTIMENT_LABELS: int = 3
    NUM_BIO_LABELS: int = 3

    # -------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------
    BATCH_SIZE: int = 16
    EVAL_BATCH_SIZE: int = 32
    LEARNING_RATE: float = 3e-5
    BILSTM_LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 4
    BILSTM_NUM_EPOCHS: int = 10
    EARLY_STOPPING_PATIENCE: int = 2
    WARMUP_RATIO: float = 0.1

    # -------------------------------------------------------------------
    # Bi-LSTM baseline hyperparameters
    # -------------------------------------------------------------------
    VOCAB_SIZE: int = 20000
    EMBEDDING_DIM: int = 128
    LSTM_UNITS: int = 64
    DROPOUT_RATE: float = 0.3

    # -------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "processed")

    ASPECT_TRAIN_FILE: str = os.path.join(PROCESSED_DATA_DIR, "aspect_train.csv")
    ASPECT_VAL_FILE: str = os.path.join(PROCESSED_DATA_DIR, "aspect_val.csv")
    ASPECT_TEST_FILE: str = os.path.join(PROCESSED_DATA_DIR, "aspect_test.csv")

    SENTIMENT_TRAIN_FILE: str = os.path.join(PROCESSED_DATA_DIR, "sentiment_train.csv")
    SENTIMENT_VAL_FILE: str = os.path.join(PROCESSED_DATA_DIR, "sentiment_val.csv")
    SENTIMENT_TEST_FILE: str = os.path.join(PROCESSED_DATA_DIR, "sentiment_test.csv")

    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
    BILSTM_MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models", "bilstm_baseline")
    ASPECT_MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models", "aspect_extraction_bert")
    SENTIMENT_MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models", "sentiment_bert")
    TOKENIZER_VOCAB_PATH: str = os.path.join(PROJECT_ROOT, "models", "bilstm_baseline", "vocab.json")

    REPORTS_DIR: str = os.path.join(PROJECT_ROOT, "reports")

    def ensure_dirs(self) -> None:
        """Create any missing directories that scripts will write to."""
        for path in [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
            self.BILSTM_MODEL_DIR,
            self.ASPECT_MODEL_DIR,
            self.SENTIMENT_MODEL_DIR,
            self.REPORTS_DIR,
        ]:
            os.makedirs(path, exist_ok=True)


# A module-level singleton for convenient importing: `from config import cfg`
cfg = Config()
