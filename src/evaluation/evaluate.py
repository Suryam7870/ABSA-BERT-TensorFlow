"""
Evaluate the aspect-extraction model, the sentiment-classification model,
and the Bi-LSTM baseline. Produces metrics (precision/recall/F1, accuracy,
confusion matrix) and plots (training curves, confusion matrix heatmap).

Usage:
    python -m src.evaluation.evaluate --model aspect
    python -m src.evaluation.evaluate --model sentiment
    python -m src.evaluation.evaluate --model baseline
    python -m src.evaluation.evaluate --model all
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402
from src.models.bilstm import Vocabulary, encode_aspect_sentiment_example  # noqa: E402
from src.preprocessing.label_alignment import align_labels_with_tokens, decode_predictions_to_words  # noqa: E402
from src.preprocessing.tokenizer import encode_sentence_pair, get_tokenizer  # noqa: E402


def plot_training_curves(history_path: str, out_path: str, title: str) -> None:
    if not os.path.exists(history_path):
        print(f"No training history found at {history_path}, skipping curve plot.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    loss_keys = [k for k in history if "loss" in k]
    for key in loss_keys:
        axes[0].plot(history[key], label=key)
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    acc_keys = [k for k in history if "acc" in k]
    if acc_keys:
        for key in acc_keys:
            axes[1].plot(history[key], label=key)
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].set_title("No accuracy metric recorded")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def plot_confusion_matrix(y_true, y_pred, labels, out_path: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


# ---------------------------------------------------------------------------
# Aspect extraction evaluation
# ---------------------------------------------------------------------------
def evaluate_aspect_model() -> None:
    from transformers import TFAutoModelForTokenClassification

    if not os.path.exists(cfg.ASPECT_TEST_FILE):
        print(f"Missing {cfg.ASPECT_TEST_FILE}. Run data preparation first.")
        return
    if not os.path.isdir(cfg.ASPECT_MODEL_DIR) or not os.listdir(cfg.ASPECT_MODEL_DIR):
        print(f"No trained aspect model found in {cfg.ASPECT_MODEL_DIR}. "
              f"Run `python -m src.training.train_aspect` first.")
        return

    print("Loading aspect extraction model ...")
    tokenizer = get_tokenizer(cfg.ASPECT_MODEL_DIR) if os.path.exists(
        os.path.join(cfg.ASPECT_MODEL_DIR, "vocab.txt")
    ) else get_tokenizer()
    model = TFAutoModelForTokenClassification.from_pretrained(cfg.ASPECT_MODEL_DIR)

    df = pd.read_csv(cfg.ASPECT_TEST_FILE).dropna(subset=["text", "tokens", "bio_labels"])
    if len(df) == 0:
        print("Aspect test set is empty. Skipping evaluation.")
        return

    all_true, all_pred = [], []
    for row in df.itertuples():
        words = json.loads(row.tokens)
        gold_labels = json.loads(row.bio_labels)
        if not words:
            continue
        aligned = align_labels_with_tokens(tokenizer, words, gold_labels)
        input_ids = tf.constant([aligned["input_ids"]])
        attention_mask = tf.constant([aligned["attention_mask"]])
        token_type_ids = tf.constant([aligned["token_type_ids"]])

        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).logits
        pred_ids = tf.argmax(logits, axis=-1).numpy()[0]

        # Re-encode to get word_ids for decoding back to word level.
        encoded = tokenizer(words, is_split_into_words=True, max_length=cfg.MAX_SEQ_LENGTH,
                             truncation=True, padding="max_length")
        word_ids = encoded.word_ids()
        pred_word_labels = decode_predictions_to_words(word_ids, pred_ids)

        n = min(len(words), len(pred_word_labels))
        all_true.append(gold_labels[:n])
        all_pred.append(pred_word_labels[:n])

    print("\n=== Aspect Extraction — seqeval classification report ===")
    report = seq_classification_report(all_true, all_pred, digits=4)
    print(report)
    macro_f1 = seq_f1_score(all_true, all_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.REPORTS_DIR, "aspect_extraction_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nMacro F1: {macro_f1:.4f}\n")

    plot_training_curves(
        os.path.join(cfg.REPORTS_DIR, "aspect_model_history.json"),
        os.path.join(cfg.REPORTS_DIR, "aspect_model_training_curves.png"),
        "Aspect Extraction (BERT)",
    )


# ---------------------------------------------------------------------------
# Sentiment classification evaluation (shared by BERT and Bi-LSTM)
# ---------------------------------------------------------------------------
def _print_and_save_classification_metrics(y_true, y_pred, labels, report_name: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)

    summary = (
        f"Accuracy: {acc:.4f}\n"
        f"Macro Precision: {precision:.4f}\n"
        f"Macro Recall: {recall:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n\n"
        f"{report}"
    )
    print(summary)

    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.REPORTS_DIR, report_name), "w", encoding="utf-8") as f:
        f.write(summary)


def evaluate_sentiment_model() -> None:
    from transformers import TFAutoModelForSequenceClassification

    if not os.path.exists(cfg.SENTIMENT_TEST_FILE):
        print(f"Missing {cfg.SENTIMENT_TEST_FILE}. Run data preparation first.")
        return
    if not os.path.isdir(cfg.SENTIMENT_MODEL_DIR) or not os.listdir(cfg.SENTIMENT_MODEL_DIR):
        print(f"No trained sentiment model found in {cfg.SENTIMENT_MODEL_DIR}. "
              f"Run `python -m src.training.train_sentiment` first.")
        return

    print("Loading sentiment classification model ...")
    tokenizer = get_tokenizer()
    model = TFAutoModelForSequenceClassification.from_pretrained(cfg.SENTIMENT_MODEL_DIR)

    df = pd.read_csv(cfg.SENTIMENT_TEST_FILE).dropna(subset=["text", "aspect", "sentiment"])
    if len(df) == 0:
        print("Sentiment test set is empty. Skipping evaluation.")
        return

    y_true = df["sentiment"].map(cfg.SENTIMENT_TO_ID).values
    y_pred = []
    for row in df.itertuples():
        encoded = encode_sentence_pair(str(row.text), str(row.aspect))
        logits = model(
            input_ids=tf.constant([encoded["input_ids"]]),
            attention_mask=tf.constant([encoded["attention_mask"]]),
            token_type_ids=tf.constant([encoded["token_type_ids"]]),
        ).logits
        pred = int(tf.argmax(logits, axis=-1).numpy()[0])
        y_pred.append(pred)

    labels = cfg.SENTIMENT_LABELS
    print("\n=== Sentiment Classification (BERT) ===")
    _print_and_save_classification_metrics(y_true, y_pred, labels, "sentiment_bert_report.txt")
    plot_confusion_matrix(
        y_true, y_pred, list(range(len(labels))),
        os.path.join(cfg.REPORTS_DIR, "sentiment_bert_confusion_matrix.png"),
        "Sentiment Classification (BERT) — Confusion Matrix",
    )
    plot_training_curves(
        os.path.join(cfg.REPORTS_DIR, "sentiment_model_history.json"),
        os.path.join(cfg.REPORTS_DIR, "sentiment_model_training_curves.png"),
        "Sentiment Classification (BERT)",
    )


def evaluate_baseline_model() -> None:
    if not os.path.exists(cfg.SENTIMENT_TEST_FILE):
        print(f"Missing {cfg.SENTIMENT_TEST_FILE}. Run data preparation first.")
        return
    model_path = os.path.join(cfg.BILSTM_MODEL_DIR, "bilstm_model.keras")
    if not os.path.exists(model_path) or not os.path.exists(cfg.TOKENIZER_VOCAB_PATH):
        print(f"No trained baseline model found in {cfg.BILSTM_MODEL_DIR}. "
              f"Run `python -m src.training.train_baseline` first.")
        return

    print("Loading Bi-LSTM baseline model ...")
    model = tf.keras.models.load_model(model_path)
    vocab = Vocabulary.load(cfg.TOKENIZER_VOCAB_PATH)

    df = pd.read_csv(cfg.SENTIMENT_TEST_FILE).dropna(subset=["text", "aspect", "sentiment"])
    if len(df) == 0:
        print("Sentiment test set is empty. Skipping evaluation.")
        return

    X = np.array(
        [
            encode_aspect_sentiment_example(vocab, row.text, row.aspect, cfg.MAX_SEQ_LENGTH)
            for row in df.itertuples()
        ]
    )
    y_true = df["sentiment"].map(cfg.SENTIMENT_TO_ID).values
    y_pred = np.argmax(model.predict(X, verbose=0), axis=-1)

    labels = cfg.SENTIMENT_LABELS
    print("\n=== Sentiment Classification (Bi-LSTM Baseline) ===")
    _print_and_save_classification_metrics(y_true, y_pred, labels, "sentiment_bilstm_report.txt")
    plot_confusion_matrix(
        y_true, y_pred, list(range(len(labels))),
        os.path.join(cfg.REPORTS_DIR, "sentiment_bilstm_confusion_matrix.png"),
        "Sentiment Classification (Bi-LSTM) — Confusion Matrix",
    )
    plot_training_curves(
        os.path.join(cfg.REPORTS_DIR, "bilstm_history.json"),
        os.path.join(cfg.REPORTS_DIR, "bilstm_training_curves.png"),
        "Bi-LSTM Baseline",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ABSA models.")
    parser.add_argument(
        "--model",
        choices=["aspect", "sentiment", "baseline", "all"],
        default="all",
    )
    args = parser.parse_args()

    cfg.ensure_dirs()

    if args.model in ("aspect", "all"):
        evaluate_aspect_model()
    if args.model in ("sentiment", "all"):
        evaluate_sentiment_model()
    if args.model in ("baseline", "all"):
        evaluate_baseline_model()


if __name__ == "__main__":
    main()
