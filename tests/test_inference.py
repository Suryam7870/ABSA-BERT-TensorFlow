"""
Unit tests for the inference pipeline (src/inference/predict.py).

These tests avoid requiring fully trained models where possible:
  - `test_predict_absa_raises_without_trained_models` checks the graceful
    error path on a fresh clone (no models/ contents yet).
  - `test_predict_absa_output_format` mocks the two model-loading stages so
    the orchestration logic and output schema can be verified without
    running real BERT inference.

Run with:
    pytest tests/test_inference.py -v
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg  # noqa: E402
from src.inference import predict as predict_module  # noqa: E402
from src.inference.predict import ModelsNotTrainedError, overall_sentiment  # noqa: E402


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure the lru_cache on model loaders doesn't leak across tests."""
    predict_module.clear_model_cache()
    yield
    predict_module.clear_model_cache()


def test_predict_absa_rejects_empty_text():
    with pytest.raises(ValueError):
        predict_module.predict_absa("")
    with pytest.raises(ValueError):
        predict_module.predict_absa("   ")


def test_predict_absa_raises_without_trained_models():
    """
    On a fresh clone (before any training has been run), models/ is empty,
    so predict_absa() must fail with a clear, actionable error rather than
    crashing unexpectedly.
    """
    if os.path.isdir(cfg.ASPECT_MODEL_DIR) and os.listdir(cfg.ASPECT_MODEL_DIR):
        pytest.skip("An aspect model is already present; skipping 'not trained' check.")

    with pytest.raises(ModelsNotTrainedError):
        predict_module.predict_absa("The food was great but the service was slow.")


def test_predict_absa_output_format(monkeypatch):
    """Mock both pipeline stages to verify the output schema and ordering."""

    def fake_extract_aspects(text: str):
        return ["food", "service"]

    def fake_classify_sentiment(text: str, aspect: str):
        sentiment_by_aspect = {"food": "positive", "service": "negative"}
        return {
            "aspect": aspect,
            "sentiment": sentiment_by_aspect[aspect],
            "confidence": 0.9,
        }

    monkeypatch.setattr(predict_module, "extract_aspects", fake_extract_aspects)
    monkeypatch.setattr(predict_module, "classify_sentiment", fake_classify_sentiment)

    result = predict_module.predict_absa(
        "The food was amazing but the service was extremely slow."
    )

    assert isinstance(result, list)
    assert len(result) == 2

    for item in result:
        assert set(item.keys()) == {"aspect", "sentiment", "confidence"}
        assert item["sentiment"] in cfg.SENTIMENT_LABELS
        assert 0.0 <= item["confidence"] <= 1.0

    assert result[0]["aspect"] == "food"
    assert result[0]["sentiment"] == "positive"
    assert result[1]["aspect"] == "service"
    assert result[1]["sentiment"] == "negative"


def test_overall_sentiment_all_positive():
    results = [{"aspect": "food", "sentiment": "positive", "confidence": 0.9}]
    assert overall_sentiment(results) == "positive"


def test_overall_sentiment_mixed():
    results = [
        {"aspect": "food", "sentiment": "positive", "confidence": 0.9},
        {"aspect": "service", "sentiment": "negative", "confidence": 0.8},
    ]
    assert overall_sentiment(results) == "mixed"


def test_overall_sentiment_no_aspects():
    assert overall_sentiment([]) == "no aspects found"


def test_overall_sentiment_all_neutral():
    results = [{"aspect": "price", "sentiment": "neutral", "confidence": 0.6}]
    assert overall_sentiment(results) == "neutral"
