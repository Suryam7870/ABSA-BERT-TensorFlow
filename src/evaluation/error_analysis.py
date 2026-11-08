"""
Error analysis for the sentiment classification model.

Runs the full inference pipeline (aspect extraction + sentiment
classification) over the sentiment test set plus a hand-picked set of
linguistically tricky sentences (negation, contrast, mixed sentiment) and
reports which aspect/sentiment pairs were predicted incorrectly.

Usage:
    python -m src.evaluation.error_analysis
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402
from src.inference.predict import predict_absa  # noqa: E402

# Hand-picked, linguistically challenging sentences that stress-test negation
# and contrastive-conjunction handling. These are NOT training data — they
# exist purely to probe model behavior after training.
CHALLENGE_SENTENCES = [
    "The food wasn't bad.",
    "The service was not good.",
    "The food was good but expensive.",
    "The food was excellent and the service was terrible.",
    "I wouldn't say the battery life is great, but it's not terrible either.",
    "The camera quality is excellent, but the battery life is disappointing.",
]


def run_challenge_set() -> None:
    print("=" * 80)
    print("CHALLENGE SET — negation / contrast / multi-aspect sentences")
    print("=" * 80)
    for sentence in CHALLENGE_SENTENCES:
        result = predict_absa(sentence)
        print(f"\nInput: {sentence}")
        if not result:
            print("  -> No aspects detected.")
        for item in result:
            print(f"  - aspect='{item['aspect']}' -> sentiment={item['sentiment']} "
                  f"(confidence={item['confidence']:.3f})")


def run_test_set_error_analysis(max_examples: int = 30) -> None:
    if not os.path.exists(cfg.SENTIMENT_TEST_FILE):
        print(f"Missing {cfg.SENTIMENT_TEST_FILE}. Run data preparation and training first.")
        return

    df = pd.read_csv(cfg.SENTIMENT_TEST_FILE).dropna(subset=["text", "aspect", "sentiment"])
    if len(df) == 0:
        print("Sentiment test set is empty.")
        return

    print("\n" + "=" * 80)
    print(f"TEST SET ERROR ANALYSIS (showing up to {max_examples} sentences)")
    print("=" * 80)

    errors = []
    seen_texts = set()
    for row in df.itertuples():
        if row.text in seen_texts:
            continue
        seen_texts.add(row.text)
        if len(seen_texts) > max_examples:
            break

        gold_for_text = df[df["text"] == row.text][["aspect", "sentiment"]]
        gold_map = {a.lower(): s for a, s in gold_for_text.values}

        predictions = predict_absa(row.text)
        for pred in predictions:
            gold_sentiment = gold_map.get(pred["aspect"].lower())
            if gold_sentiment is not None and gold_sentiment != pred["sentiment"]:
                errors.append(
                    {
                        "text": row.text,
                        "aspect": pred["aspect"],
                        "gold": gold_sentiment,
                        "predicted": pred["sentiment"],
                        "confidence": pred["confidence"],
                    }
                )

    if not errors:
        print("No misclassifications found in the sampled examples (or models "
              "have not been trained yet).")
        return

    error_df = pd.DataFrame(errors)
    print(error_df.to_string(index=False))

    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.REPORTS_DIR, "error_analysis.csv")
    error_df.to_csv(out_path, index=False)
    print(f"\nSaved full error analysis to {out_path}")


if __name__ == "__main__":
    run_challenge_set()
    run_test_set_error_analysis()
