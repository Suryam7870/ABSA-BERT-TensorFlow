# Dataset

This project uses the **SemEval-2014 Task 4: Aspect Based Sentiment
Analysis** dataset (Restaurant Reviews domain), the standard benchmark
dataset for aspect-based sentiment analysis.

The raw dataset is **not included** in this repository, because it is
distributed by the SemEval task organizers under a research-use license
that does not permit redistribution.

## Option A — Automatic download (recommended)

```bash
python -m src.data.download_data
```

This script attempts to fetch a mirror of the dataset from the Hugging Face
Hub (`alexcadillon/SemEval2014Task4`, `restaurants` config) using the
`datasets` library and writes JSON-lines files into `data/raw/`.

If this succeeds, continue with:

```bash
python -m src.data.prepare_data
```

## Option B — Manual download

If the automatic download fails (no internet access, mirror unavailable,
etc.), download the data manually:

1. Visit the official SemEval-2014 Task 4 page:
   https://alt.qcri.org/semeval2014/task4/
2. Download the Restaurant Reviews training and test files, e.g.:
   - `Restaurants_Train_v2.xml`
   - `Restaurants_Test_Gold.xml`
3. Place both files directly inside this `data/raw/` folder:
   ```
   data/raw/Restaurants_Train_v2.xml
   data/raw/Restaurants_Test_Gold.xml
   ```
4. Run:
   ```bash
   python -m src.data.prepare_data --input-format xml
   ```

## What `prepare_data.py` produces

Running the preparation script converts the raw XML/JSONL files into six CSV
files inside `data/processed/`:

| File                       | Used by                | Columns                        |
|-----------------------------|------------------------|---------------------------------|
| `aspect_train.csv`          | Aspect extraction (BERT) | `text`, `tokens`, `bio_labels` |
| `aspect_val.csv`            | Aspect extraction (BERT) | `text`, `tokens`, `bio_labels` |
| `aspect_test.csv`           | Aspect extraction (BERT) | `text`, `tokens`, `bio_labels` |
| `sentiment_train.csv`       | Sentiment classification (BERT & Bi-LSTM) | `text`, `aspect`, `sentiment` |
| `sentiment_val.csv`         | Sentiment classification (BERT & Bi-LSTM) | `text`, `aspect`, `sentiment` |
| `sentiment_test.csv`        | Sentiment classification (BERT & Bi-LSTM) | `text`, `aspect`, `sentiment` |

The preparation script also:
- Drops records with missing/empty text or aspect terms
- Drops records with polarities outside `{positive, negative, neutral}`
  (e.g. the `conflict` label present in the original SemEval data, since
  this project targets a clean 3-class setting)
- Drops duplicate records
- Aligns aspect terms to whitespace tokens to build BIO labels
- Splits into train (80%) / validation (10%) / test (10%), stratified by
  sentiment label where possible

## Using a different dataset

If you substitute a different ABSA dataset, make sure it can be expressed as
records of the form:

```json
{"text": "The battery life is excellent.", "aspects": [{"term": "battery life", "polarity": "positive"}]}
```

and adapt `parse_xml_file` / `parse_jsonl_file` in
`src/data/prepare_data.py` accordingly. Do not fabricate labels — only use
data that was genuinely annotated.
