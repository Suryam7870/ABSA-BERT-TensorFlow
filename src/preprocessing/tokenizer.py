"""
Thin wrapper around the Hugging Face tokenizer used throughout the project.

Centralizing tokenizer creation here ensures every script (training,
evaluation, inference) tokenizes text identically.
"""

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import cfg


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = None) -> PreTrainedTokenizerBase:
    """
    Load (and cache) the pretrained BERT tokenizer.

    Args:
        model_name: Optional override of the model name in config.py.

    Returns:
        A Hugging Face fast tokenizer instance.
    """
    name = model_name or cfg.BERT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return tokenizer


def encode_sentence_pair(text_a: str, text_b: str, max_length: int = None) -> dict:
    """
    Encode a `[CLS] text_a [SEP] text_b [SEP]` style input, used for aspect
    sentiment classification (review text + aspect term).

    Returns a dict with input_ids, attention_mask, and token_type_ids as
    Python lists (ready to be batched into tensors by the caller).
    """
    tokenizer = get_tokenizer()
    max_len = max_length or cfg.MAX_SEQ_LENGTH
    encoded = tokenizer(
        text_a,
        text_b,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_token_type_ids=True,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded["token_type_ids"],
    }


def encode_single_sentence(text: str, max_length: int = None) -> dict:
    """Encode a single sentence (used for token classification / aspect extraction)."""
    tokenizer = get_tokenizer()
    max_len = max_length or cfg.MAX_SEQ_LENGTH
    encoded = tokenizer(
        text.split(),
        max_length=max_len,
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
        return_token_type_ids=True,
    )
    return encoded
