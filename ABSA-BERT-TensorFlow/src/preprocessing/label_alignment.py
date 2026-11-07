"""
Word-level BIO label <-> BERT subword token alignment.

BERT's WordPiece tokenizer splits words such as "disappointing" into several
subword pieces ("disappoint", "##ing"). The gold labels in the processed CSVs
are at the *word* level, so they must be re-aligned to the *subword* level
before being fed to the token-classification model.

Convention used here (the standard approach, e.g. in the Hugging Face
`run_ner` examples):
  - The first subword of a word gets the word's original label.
  - Subsequent subwords of the same word get the IGNORE_LABEL_ID (-100),
    so they do not contribute to the loss.
  - Special tokens ([CLS], [SEP], [PAD]) also get IGNORE_LABEL_ID.
"""

from typing import List

from transformers import PreTrainedTokenizerBase

from config import cfg


def align_labels_with_tokens(
    tokenizer: PreTrainedTokenizerBase,
    words: List[str],
    word_labels: List[str],
    max_length: int = None,
) -> dict:
    """
    Tokenize a pre-tokenized (whitespace-split) sentence and align its
    word-level BIO labels to the resulting subword tokens.

    Args:
        tokenizer: A fast Hugging Face tokenizer (must support `is_split_into_words`).
        words: List of whitespace-separated words in the sentence.
        word_labels: List of BIO label strings, one per word (same length as `words`).
        max_length: Optional override of the max sequence length.

    Returns:
        dict with keys: input_ids, attention_mask, token_type_ids, labels
        (all as plain Python lists, `labels` uses cfg.IGNORE_LABEL_ID for
        tokens that should not contribute to the loss).
    """
    if len(words) != len(word_labels):
        raise ValueError(
            f"words and word_labels must have the same length "
            f"(got {len(words)} vs {len(word_labels)})"
        )

    max_len = max_length or cfg.MAX_SEQ_LENGTH
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_token_type_ids=True,
    )

    word_ids = encoded.word_ids()  # None for special tokens
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token: [CLS], [SEP], or [PAD]
            aligned_labels.append(cfg.IGNORE_LABEL_ID)
        elif word_id != previous_word_id:
            # First subword token of a new word: use its real label.
            label = word_labels[word_id]
            aligned_labels.append(cfg.LABEL_TO_ID.get(label, cfg.LABEL_TO_ID["O"]))
        else:
            # A continuation subword token of the same word: ignore in loss.
            aligned_labels.append(cfg.IGNORE_LABEL_ID)
        previous_word_id = word_id

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded["token_type_ids"],
        "labels": aligned_labels,
    }


def decode_predictions_to_words(
    word_ids: List, predicted_label_ids: List[int], id_to_label: dict = None
) -> List[str]:
    """
    Collapse subword-level predictions back to word-level predictions by
    taking the prediction of each word's first subword token.

    Args:
        word_ids: The `word_ids()` list returned by the tokenizer for this example.
        predicted_label_ids: Model's predicted label id for every token position.
        id_to_label: Optional override of the id->label mapping.

    Returns:
        A list of predicted BIO label strings, one per original word.
    """
    mapping = id_to_label or cfg.ID_TO_LABEL
    word_level_labels = []
    seen_words = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_words:
            continue
        seen_words.add(word_id)
        label_id = predicted_label_ids[idx]
        word_level_labels.append(mapping.get(label_id, "O"))
    return word_level_labels
