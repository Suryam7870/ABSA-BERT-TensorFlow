"""
Convert raw ABSA data (SemEval-2014 Task 4 style) into the flat CSV formats
used by the aspect-extraction and sentiment-classification training scripts.

Supported raw formats:
  1. Official SemEval XML files placed in data/raw/ (e.g. Restaurants_Train_v2.xml)
  2. JSON-lines files produced by src/data/download_data.py (data/raw/*.jsonl)

Output (written to data/processed/):
  aspect_train.csv / aspect_val.csv / aspect_test.csv
      columns: text, tokens (JSON list), bio_labels (JSON list)

  sentiment_train.csv / sentiment_val.csv / sentiment_test.csv
      columns: text, aspect, sentiment

Handles: missing values, invalid records, duplicates, and train/val/test
splitting. Prints a summary of how many records were kept/dropped.

Usage:
    python -m src.data.prepare_data
    python -m src.data.prepare_data --input-format xml
"""

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import cfg  # noqa: E402

VALID_POLARITIES = {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# Raw file parsing
# ---------------------------------------------------------------------------
def parse_xml_file(path: str) -> List[Dict]:
    """Parse an official SemEval-2014 Task 4 XML file into a list of records."""
    records = []
    tree = ET.parse(path)
    root = tree.getroot()
    for sentence in root.findall("sentence"):
        text_el = sentence.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text.strip()
        aspects = []
        aspect_terms_el = sentence.find("aspectTerms")
        if aspect_terms_el is not None:
            for term_el in aspect_terms_el.findall("aspectTerm"):
                term = term_el.get("term")
                polarity = term_el.get("polarity")
                if term is None or polarity is None:
                    continue
                aspects.append({"term": term.strip(), "polarity": polarity.strip().lower()})
        records.append({"text": text, "aspects": aspects})
    return records


def _extract_aspects_from_generic_dict(row: dict) -> List[Dict]:
    """Try several plausible key names used by different dataset mirrors."""
    aspects = []
    candidates = None
    for key in ("aspectTerms", "aspect_terms", "aspects", "opinions", "Opinions"):
        if key in row and row[key]:
            candidates = row[key]
            break
    if not candidates:
        return aspects

    for item in candidates:
        if not isinstance(item, dict):
            continue
        term = (
            item.get("term")
            or item.get("aspect")
            or item.get("target")
            or item.get("category")
        )
        polarity = item.get("polarity") or item.get("sentiment") or item.get("label")
        if term is None or polarity is None:
            continue
        aspects.append({"term": str(term).strip(), "polarity": str(polarity).strip().lower()})
    return aspects


def parse_jsonl_file(path: str) -> List[Dict]:
    """Parse a JSON-lines file (from download_data.py) into a list of records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("raw_words") or row.get("sentence")
            if not text:
                continue
            aspects = _extract_aspects_from_generic_dict(row)
            records.append({"text": str(text).strip(), "aspects": aspects})
    return records


def load_raw_records(input_format: Optional[str] = None) -> List[Dict]:
    """Load every raw file found in data/raw/ and merge into one record list."""
    records: List[Dict] = []

    xml_files = sorted(glob.glob(os.path.join(cfg.RAW_DATA_DIR, "*.xml")))
    jsonl_files = sorted(glob.glob(os.path.join(cfg.RAW_DATA_DIR, "*.jsonl")))

    if input_format == "xml":
        jsonl_files = []
    elif input_format == "jsonl":
        xml_files = []

    for path in xml_files:
        print(f"Parsing XML file: {path}")
        records.extend(parse_xml_file(path))

    for path in jsonl_files:
        print(f"Parsing JSONL file: {path}")
        records.extend(parse_jsonl_file(path))

    return records


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_records(records: List[Dict]) -> List[Dict]:
    """
    Drop records with missing text, invalid polarities, or duplicated
    (text, aspects) pairs. Returns the cleaned list.
    """
    seen_texts = set()
    cleaned = []
    dropped_missing = 0
    dropped_invalid_polarity = 0
    dropped_duplicate = 0

    for rec in records:
        text = rec.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            dropped_missing += 1
            continue

        text = " ".join(text.split())  # normalize whitespace

        valid_aspects = []
        for aspect in rec.get("aspects", []):
            term = aspect.get("term", "").strip()
            polarity = aspect.get("polarity", "").strip().lower()
            if not term:
                dropped_missing += 1
                continue
            if polarity not in VALID_POLARITIES:
                # "conflict" polarity (used in some SemEval sets) and unknown
                # labels are dropped since this project targets a 3-class
                # (positive/negative/neutral) setting.
                dropped_invalid_polarity += 1
                continue
            valid_aspects.append({"term": term, "polarity": polarity})

        key = (text, tuple(sorted((a["term"], a["polarity"]) for a in valid_aspects)))
        if key in seen_texts:
            dropped_duplicate += 1
            continue
        seen_texts.add(key)

        cleaned.append({"text": text, "aspects": valid_aspects})

    print(
        f"Cleaning summary: kept={len(cleaned)}, "
        f"dropped_missing={dropped_missing}, "
        f"dropped_invalid_polarity={dropped_invalid_polarity}, "
        f"dropped_duplicate={dropped_duplicate}"
    )
    return cleaned


# ---------------------------------------------------------------------------
# BIO tagging for aspect extraction
# ---------------------------------------------------------------------------
def build_bio_labels(text: str, aspects: List[Dict]) -> Optional[List[str]]:
    """
    Produce word-level BIO labels for a sentence given its aspect terms.

    A simple, robust strategy: tokenize on whitespace, then for each aspect
    term look for a contiguous, case-insensitive token match inside the
    sentence tokens and mark it B-ASPECT / I-ASPECT. Aspect terms whose words
    cannot be located are skipped (this can happen due to punctuation
    differences) and logged by the caller.
    """
    tokens = text.split()
    labels = ["O"] * len(tokens)
    lowered_tokens = [t.strip(".,!?;:\"'()").lower() for t in tokens]

    for aspect in aspects:
        term_tokens = [t.strip(".,!?;:\"'()").lower() for t in aspect["term"].split()]
        if not term_tokens:
            continue
        term_len = len(term_tokens)
        match_found = False
        for start in range(len(lowered_tokens) - term_len + 1):
            if lowered_tokens[start:start + term_len] == term_tokens:
                labels[start] = "B-ASPECT"
                for i in range(start + 1, start + term_len):
                    labels[i] = "I-ASPECT"
                match_found = True
                break
        if not match_found:
            # Could not align this aspect term to tokens; skip silently.
            continue

    return tokens, labels


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def build_aspect_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        tokens, labels = build_bio_labels(rec["text"], rec["aspects"])
        rows.append(
            {
                "text": rec["text"],
                "tokens": json.dumps(tokens),
                "bio_labels": json.dumps(labels),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)


def build_sentiment_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for aspect in rec["aspects"]:
            rows.append(
                {
                    "text": rec["text"],
                    "aspect": aspect["term"],
                    "sentiment": aspect["polarity"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["text", "aspect", "sentiment"]).reset_index(drop=True)


def split_and_save(df: pd.DataFrame, train_path: str, val_path: str, test_path: str,
                    stratify_col: Optional[str] = None) -> None:
    if df.empty:
        print(f"WARNING: no data to write for {train_path} — skipping split.")
        for p in (train_path, val_path, test_path):
            pd.DataFrame(columns=df.columns).to_csv(p, index=False)
        return

    stratify = df[stratify_col] if stratify_col and df[stratify_col].nunique() > 1 else None
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=cfg.RANDOM_SEED, stratify=stratify
    )
    stratify_temp = temp_df[stratify_col] if stratify_col and temp_df[stratify_col].nunique() > 1 else None
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=cfg.RANDOM_SEED, stratify=stratify_temp
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")


def main(input_format: Optional[str] = None) -> None:
    cfg.ensure_dirs()

    records = load_raw_records(input_format=input_format)
    if not records:
        print(
            "No raw data files found in data/raw/. Run "
            "`python -m src.data.download_data` first, or see data/README.md "
            "for manual download instructions."
        )
        sys.exit(1)

    records = clean_records(records)
    if cfg.DEBUG_MODE:
        print(f"DEBUG_MODE is True -> truncating to {cfg.DEBUG_SAMPLE_SIZE} records")
        records = records[: cfg.DEBUG_SAMPLE_SIZE]

    print("\nBuilding aspect-extraction dataset (BIO tagging) ...")
    aspect_df = build_aspect_dataframe(records)
    print("Splitting aspect-extraction dataset ...")
    split_and_save(
        aspect_df, cfg.ASPECT_TRAIN_FILE, cfg.ASPECT_VAL_FILE, cfg.ASPECT_TEST_FILE
    )

    print("\nBuilding sentiment-classification dataset ...")
    sentiment_df = build_sentiment_dataframe(records)
    print("Splitting sentiment-classification dataset ...")
    split_and_save(
        sentiment_df,
        cfg.SENTIMENT_TRAIN_FILE,
        cfg.SENTIMENT_VAL_FILE,
        cfg.SENTIMENT_TEST_FILE,
        stratify_col="sentiment",
    )

    print("\nDone. Processed files written to:", cfg.PROCESSED_DATA_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ABSA training data.")
    parser.add_argument(
        "--input-format",
        choices=["xml", "jsonl"],
        default=None,
        help="Restrict parsing to a single raw format. Default: use all files found.",
    )
    args = parser.parse_args()
    main(input_format=args.input_format)
