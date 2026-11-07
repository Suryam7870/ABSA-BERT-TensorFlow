"""
Download the SemEval-2014 Task 4 (Restaurants) Aspect-Based Sentiment Analysis
dataset.

The official SemEval-2014 Task 4 data is distributed by the task organizers
under a research-use license and cannot be legally redistributed inside this
repository. This script instead tries to fetch a community mirror of the
dataset that is hosted on the Hugging Face Hub (`datasets` library) and
converts it into the flat CSV format expected by `prepare_data.py`.

If the download fails (no internet access, mirror unavailable, etc.) the
script prints clear instructions for downloading the data manually and
placing it in `data/raw/`. See `data/README.md` for the full explanation.

Usage:
    python -m src.data.download_data
"""

import json
import os
import sys

# Allow running this file both as a module (`python -m src.data.download_data`)
# and directly (`python src/data/download_data.py`).
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402

HF_DATASET_ID = "alexcadillon/SemEval2014Task4"
HF_DATASET_CONFIG = "restaurants"

MANUAL_INSTRUCTIONS = """
================================================================================
Automatic download was not successful.

The SemEval-2014 Task 4 Restaurant Reviews dataset must be obtained manually:

  1. Visit: https://alt.qcri.org/semeval2014/task4/
     (Official SemEval-2014 Task 4: Aspect Based Sentiment Analysis)

  2. Register / download the "Restaurants" training and test XML files, e.g.:
       Restaurants_Train_v2.xml
       Restaurants_Test_Gold.xml

  3. Place the downloaded XML files in:
       data/raw/Restaurants_Train_v2.xml
       data/raw/Restaurants_Test_Gold.xml

  4. Re-run this script, or run:
       python -m src.data.prepare_data --input-format xml

If you cannot access the official dataset, any ABSA dataset in the same
format (sentence + list of aspect terms + polarity) can be substituted -
just adapt `src/data/prepare_data.py` accordingly. Do NOT fabricate data.

Note on the Hugging Face mirror used by this script:
  The 'datasets' library version 3.0+ removed support for dataset repos
  that ship a Python loading script (which 'alexcadillon/SemEval2014Task4'
  uses), even with trust_remote_code=True. If you want to retry the
  automatic path, downgrade the library first:

      pip install "datasets<3.0"

  then re-run:

      python -m src.data.download_data
================================================================================
"""


def try_huggingface_download() -> bool:
    """
    Attempt to download the dataset via the `datasets` library from the
    Hugging Face Hub and write it out as raw JSON lines under data/raw/.

    Returns:
        True if the download succeeded, False otherwise.
    """
    try:
        import datasets as datasets_lib
        from datasets import load_dataset
    except ImportError:
        print("The `datasets` package is not installed. Run: pip install datasets")
        return False

    datasets_major_version = int(datasets_lib.__version__.split(".")[0])
    if datasets_major_version >= 3:
        print(
            f"Installed `datasets` version is {datasets_lib.__version__}. "
            f"Versions >= 3.0 removed support for script-based dataset repos "
            f"(the mirror this project uses is script-based), so the "
            f"automatic download cannot work with this version. "
            f"Run `pip install \"datasets<3.0\"` to use the automatic path, "
            f"or follow the manual download instructions below."
        )
        return False

    try:
        print(f"Attempting to download '{HF_DATASET_ID}' ({HF_DATASET_CONFIG}) "
              f"from the Hugging Face Hub ...")
        dataset = load_dataset(HF_DATASET_ID, HF_DATASET_CONFIG, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - we want to catch any download failure
        print(f"Download failed: {exc}")
        return False

    os.makedirs(cfg.RAW_DATA_DIR, exist_ok=True)
    for split_name, split_data in dataset.items():
        out_path = os.path.join(cfg.RAW_DATA_DIR, f"restaurants_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in split_data:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(split_data)} examples to {out_path}")

    return True


def main() -> None:
    cfg.ensure_dirs()
    success = try_huggingface_download()
    if not success:
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)
    print("\nDataset download complete. Next step:")
    print("  python -m src.data.prepare_data")


if __name__ == "__main__":
    main()
