"""
Convenience script that runs the full pipeline end-to-end:

    1. Download raw data (if not already present)
    2. Prepare / preprocess data into train/val/test CSVs
    3. Train the Bi-LSTM baseline
    4. Train the BERT aspect-extraction model
    5. Train the BERT sentiment-classification model
    6. Run evaluation on all three models

Usage:
    python run_training.py
    python run_training.py --skip-download
    python run_training.py --only prepare,baseline
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import cfg  # noqa: E402

STEPS = ["download", "prepare", "baseline", "aspect", "sentiment", "evaluate"]


def run_step(step: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {step}")
    print("=" * 80)

    if step == "download":
        from src.data.download_data import main as download_main
        download_main()
    elif step == "prepare":
        from src.data.prepare_data import main as prepare_main
        prepare_main()
    elif step == "baseline":
        from src.training.train_baseline import main as baseline_main
        baseline_main()
    elif step == "aspect":
        from src.training.train_aspect import main as aspect_main
        aspect_main()
    elif step == "sentiment":
        from src.training.train_sentiment import main as sentiment_main
        sentiment_main()
    elif step == "evaluate":
        from src.evaluation.evaluate import main as evaluate_main
        sys.argv = ["evaluate.py", "--model", "all"]
        evaluate_main()
    else:
        raise ValueError(f"Unknown step: {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ABSA training pipeline.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated subset of steps to run. Available: {', '.join(STEPS)}",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step (use existing files in data/raw/).",
    )
    args = parser.parse_args()

    cfg.ensure_dirs()

    if args.only:
        steps_to_run = [s.strip() for s in args.only.split(",") if s.strip()]
        invalid = set(steps_to_run) - set(STEPS)
        if invalid:
            raise ValueError(f"Unknown step(s): {invalid}. Available: {STEPS}")
    else:
        steps_to_run = list(STEPS)
        if args.skip_download and "download" in steps_to_run:
            steps_to_run.remove("download")

    print(f"DEBUG_MODE = {cfg.DEBUG_MODE} "
          f"({'small sample, fast run' if cfg.DEBUG_MODE else 'full dataset, full training'})")
    print(f"Steps to run: {steps_to_run}")

    for step in steps_to_run:
        run_step(step)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
