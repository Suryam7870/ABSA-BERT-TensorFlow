# Aspect-Based Sentiment Analysis using BERT and TensorFlow

A complete, runnable Aspect-Based Sentiment Analysis (ABSA) system built with
**BERT + TensorFlow/Keras**. Given a review, it extracts the aspects being
discussed (e.g. *food*, *service*, *battery life*) and predicts the sentiment
(*positive* / *negative* / *neutral*) expressed toward each one, with a
Streamlit web interface for interactive use.

> ⚠️ This repository ships **code**, not pretrained model weights. You must
> run the data preparation and training scripts yourself before the
> inference pipeline or the Streamlit app will produce predictions. See
> [Training](#12-training) below.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [What is ABSA?](#3-what-is-absa)
4. [Features](#4-features)
5. [Architecture](#5-architecture)
6. [Dataset](#6-dataset)
7. [Tech Stack](#7-tech-stack)
8. [Project Structure](#8-project-structure)
9. [Installation](#9-installation)
10. [Dataset Setup](#10-dataset-setup)
11. [Data Preprocessing](#11-data-preprocessing)
12. [Training](#12-training)
13. [Evaluation](#13-evaluation)
14. [Inference](#14-inference)
15. [Streamlit Application](#15-streamlit-application)
16. [Example](#16-example)
17. [Results](#17-results)
18. [Limitations](#18-limitations)
19. [Future Improvements](#19-future-improvements)
20. [Author](#20-author)

---

## 1. Project Overview

This project implements a two-stage deep learning pipeline for Aspect-Based
Sentiment Analysis on the SemEval-2014 Task 4 restaurant reviews benchmark,
using BERT fine-tuned with TensorFlow/Keras via Hugging Face Transformers'
TensorFlow (`TF*`) model classes. A Bi-LSTM baseline is included for
comparison.

## 2. Problem Statement

Traditional sentiment analysis assigns a single sentiment label to an entire
document or sentence. This is insufficient when a review discusses multiple
things with different opinions, e.g.:

> "The food was amazing but the service was extremely slow."

A single sentence-level label ("mixed" or an average) loses the fact that
**food = positive** and **service = negative**. This project solves that by
predicting sentiment **per aspect**, not per sentence.

## 3. What is ABSA?

Aspect-Based Sentiment Analysis (ABSA) is the task of:

1. **Identifying aspect terms** — the specific entities/attributes being
   discussed (e.g. "battery life", "camera quality", "service").
2. **Classifying sentiment per aspect** — determining whether the opinion
   expressed toward each aspect is positive, negative, or neutral.

## 4. Features

- Two-stage BERT pipeline: aspect extraction (token classification) +
  aspect sentiment classification (sequence classification)
- Bi-LSTM baseline for direct comparison against BERT
- Correct BERT subword-to-word label alignment (with `-100` ignore labels)
- Clean data preprocessing pipeline with missing-value/duplicate handling
- Full evaluation suite: precision/recall/F1, macro/weighted F1, confusion
  matrices, and training-curve plots
- Error analysis tooling, including hand-picked negation/contrast test
  sentences
- A polished Streamlit web app with cached model loading
- Unit tests that don't require a trained model to run
- `DEBUG_MODE` toggle for fast iteration on a laptop CPU vs. full training

## 5. Architecture

```
                     ┌─────────────────────────┐
                     │       Raw Review         │
                     │  "The food was amazing   │
                     │  but the service was     │
                     │  extremely slow."        │
                     └────────────┬─────────────┘
                                  │
                 ┌────────────────▼─────────────────┐
                 │  STAGE A — Aspect Extraction      │
                 │  BERT + Token Classification      │
                 │  BIO tags: O / B-ASPECT / I-ASPECT│
                 └────────────────┬───────────────────┘
                                  │
                     aspects: ["food", "service"]
                                  │
                 ┌────────────────▼───────────────────┐
                 │ STAGE B — Aspect Sentiment          │
                 │ BERT + Sequence Classification      │
                 │ Input: [CLS] review [SEP] aspect[SEP]│
                 └────────────────┬───────────────────┘
                                  │
                 ┌────────────────▼───────────────────┐
                 │ food    -> positive (0.96)          │
                 │ service -> negative (0.94)          │
                 └──────────────────────────────────────┘
```

A separate **Bi-LSTM baseline** (`Embedding -> BiLSTM -> Dropout -> Dense ->
Softmax`) is trained on the same sentiment-classification task for
comparison purposes.

## 6. Dataset

**SemEval-2014 Task 4** (Restaurant Reviews) — the standard ABSA benchmark.
The raw dataset is **not redistributed** in this repository (license
restriction). See [`data/README.md`](data/README.md) for exact download
instructions (automatic via Hugging Face mirror, or manual from the official
SemEval site).

## 7. Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9–3.12 |
| Deep Learning | TensorFlow / Keras |
| NLP | Hugging Face Transformers (`TFAutoModelForTokenClassification`, `TFAutoModelForSequenceClassification`), BERT (`bert-base-uncased`) |
| Data | Pandas, NumPy |
| Evaluation | scikit-learn, seqeval |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Testing | pytest |

**No PyTorch is used anywhere in this project.**

## 8. Project Structure

```
ABSA-BERT-TensorFlow/
│
├── app/
│   └── streamlit_app.py          # Web interface
│
├── data/
│   ├── raw/                      # Raw downloaded data (gitignored)
│   ├── processed/                # Processed CSVs (gitignored)
│   └── README.md                 # Dataset sourcing instructions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bilstm_baseline.ipynb
│   └── 03_bert_absa.ipynb
│
├── src/
│   ├── data/
│   │   ├── download_data.py      # Fetch raw dataset
│   │   └── prepare_data.py       # Clean, BIO-tag, split into CSVs
│   │
│   ├── preprocessing/
│   │   ├── tokenizer.py          # BERT tokenizer wrapper
│   │   └── label_alignment.py    # Subword <-> word BIO alignment
│   │
│   ├── models/
│   │   ├── bilstm.py             # Bi-LSTM baseline
│   │   ├── aspect_model.py       # BERT token classification
│   │   └── sentiment_model.py    # BERT sequence classification
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_aspect.py
│   │   └── train_sentiment.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py           # Metrics + plots
│   │   └── error_analysis.py     # Inspect misclassifications
│   │
│   └── inference/
│       └── predict.py            # predict_absa(text)
│
├── models/                       # Trained models saved here (gitignored)
├── reports/                      # Evaluation outputs (plots, reports)
├── tests/
│   ├── test_preprocessing.py
│   └── test_inference.py
│
├── requirements.txt
├── README.md
├── .gitignore
├── config.py
└── run_training.py
```

## 9. Installation

Windows (PowerShell/CMD):

```bat
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 10. Dataset Setup

```bash
python -m src.data.download_data
```

If automatic download fails, follow the manual instructions printed by the
script, or read [`data/README.md`](data/README.md).

## 11. Data Preprocessing

```bash
python -m src.data.prepare_data
```

This converts the raw data into six CSV files in `data/processed/` (three
for aspect extraction, three for sentiment classification), handling missing
values, invalid labels, duplicates, and an 80/10/10 train/val/test split.
See `data/README.md` for the exact output format.

## 12. Training

Train each model independently:

```bash
python -m src.training.train_baseline    # Bi-LSTM baseline
python -m src.training.train_aspect      # BERT aspect extraction
python -m src.training.train_sentiment   # BERT sentiment classification
```

Or run the entire pipeline end-to-end:

```bash
python run_training.py
```

**`DEBUG_MODE`** (set in `config.py`) defaults to `True`, which trains on a
small (`DEBUG_SAMPLE_SIZE=200`) sample for a couple of epochs — useful for
verifying the whole pipeline runs correctly on a laptop CPU in minutes.
Set `DEBUG_MODE = False` in `config.py` for a full training run on the
complete dataset with the full number of epochs (`NUM_EPOCHS`,
`BILSTM_NUM_EPOCHS`).

Trained models are saved to `models/` (excluded from git via `.gitignore` —
see [Limitations](#18-limitations)).

## 13. Evaluation

```bash
python -m src.evaluation.evaluate --model all
```

Produces, in `reports/`:
- `aspect_extraction_report.txt` — precision/recall/F1 (seqeval, entity-level) + macro F1
- `sentiment_bert_report.txt` / `sentiment_bilstm_report.txt` — accuracy, macro/weighted F1, per-class report
- `sentiment_bert_confusion_matrix.png` / `sentiment_bilstm_confusion_matrix.png`
- `*_training_curves.png` — training vs. validation loss/accuracy

For qualitative error inspection (including negation/contrast test cases):

```bash
python -m src.evaluation.error_analysis
```

## 14. Inference

```python
from src.inference.predict import predict_absa

result = predict_absa(
    "The food was amazing but the service was very slow."
)
print(result)
# [
#     {"aspect": "food", "sentiment": "positive", "confidence": 0.96},
#     {"aspect": "service", "sentiment": "negative", "confidence": 0.94},
# ]
```

Confidence is the sentiment model's own softmax probability for the
predicted class — it is **not** a calibrated probability.

## 15. Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).
Models are loaded once per server process via `st.cache_resource`, so the
app does not reload or retrain models on every interaction.

## 16. Example

Input:
```
The camera quality is excellent, but the battery life is disappointing.
```

Output:

| Aspect | Sentiment |
|---|---|
| camera quality | Positive |
| battery life | Negative |

Overall Sentiment: **Mixed**

## 17. Results

Run the evaluation script to generate results:

```bash
python -m src.evaluation.evaluate --model all
```

This README intentionally does not report fabricated accuracy/F1 numbers.
After training and evaluating on your machine, actual metrics will be
written to `reports/*.txt` and can be pasted here.

## 18. Limitations

- Trained model weights are **not** included in this repository (by design
  — see `.gitignore`); you must train them locally before inference works.
- Aspect-to-token alignment during data preparation uses exact (case-
  insensitive) whitespace-token matching, which can miss aspect terms that
  differ from the review text due to punctuation or minor tokenization
  differences.
- The Bi-LSTM baseline uses a simple whitespace/regex tokenizer and
  randomly-initialized embeddings (no pretrained word vectors), so it is
  expected to underperform BERT — that comparison is the point of including it.
- Negation and contrastive-conjunction handling rely entirely on what BERT
  can learn from the training data; there are no hand-written negation
  rules. See `src/evaluation/error_analysis.py` for stress-test sentences.
- Confidence scores are raw softmax outputs, not calibrated probabilities.

## 19. Future Improvements

- Joint (single-model) aspect extraction + sentiment tagging instead of two
  separate BERT models
- Aspect-category classification (in addition to aspect-term extraction)
- Data augmentation for underrepresented sentiment classes
- Confidence calibration (e.g. temperature scaling)
- Support for additional domains (laptops, hotels) via multi-domain training
- ONNX / TensorFlow Lite export for faster CPU inference

## 20. Author

Generated as a complete, runnable ABSA reference project. Contributions and
issues are welcome — adapt freely for your own use case.
