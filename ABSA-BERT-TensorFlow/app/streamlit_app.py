"""
Streamlit web application for the Aspect-Based Sentiment Analysis project.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg  # noqa: E402
from src.inference.predict import (  # noqa: E402
    ModelsNotTrainedError,
    overall_sentiment,
    predict_absa,
)

st.set_page_config(
    page_title="Aspect-Based Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

SENTIMENT_COLORS = {
    "positive": "#1a7f37",
    "negative": "#cf222e",
    "neutral": "#6e7781",
}

SENTIMENT_ICONS = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral": "⚪",
}


@st.cache_resource(show_spinner=False)
def warm_up_models():
    """
    Trigger model loading once per Streamlit server process and cache the
    result, so models are not reloaded/retrained on every user interaction.
    """
    from src.inference.predict import _load_aspect_model, _load_sentiment_model

    _load_aspect_model()
    _load_sentiment_model()
    return True


def render_results(text: str, results: list) -> None:
    if not results:
        st.warning("No aspects were detected in this review.")
        return

    df = pd.DataFrame(results)
    df["Sentiment"] = df["sentiment"].apply(
        lambda s: f"{SENTIMENT_ICONS.get(s, '')} {s.capitalize()}"
    )
    df["Confidence"] = df["confidence"].apply(lambda c: f"{c:.1%}")
    display_df = df.rename(columns={"aspect": "Aspect"})[["Aspect", "Sentiment", "Confidence"]]

    st.subheader("Results")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    overall = overall_sentiment(results)
    st.subheader("Overall Sentiment")
    color = SENTIMENT_COLORS.get(overall, "#444444")
    st.markdown(
        f"<h3 style='color:{color}; margin-top:-10px;'>{overall.capitalize()}</h3>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("🧠 Aspect-Based Sentiment Analyzer")
    st.caption(
        "Enter a product or restaurant review below. The app extracts "
        "individual aspects (e.g. *food*, *service*, *battery life*) and "
        "predicts the sentiment expressed toward each one, using a "
        "BERT + TensorFlow two-stage pipeline."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "This app runs two fine-tuned BERT models:\n\n"
            "1. **Aspect extraction** — BIO token classification\n"
            "2. **Sentiment classification** — per-aspect polarity\n\n"
            "Models must be trained locally before this app can produce "
            "predictions. See the project README for training instructions."
        )
        st.divider()
        st.write(f"Base model: `{cfg.BERT_MODEL_NAME}`")
        st.write(f"Max sequence length: `{cfg.MAX_SEQ_LENGTH}`")

    example = "The food was amazing but the service was extremely slow."
    text = st.text_area("Enter your review:", value="", placeholder=example, height=120)

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    with col2:
        use_example_clicked = st.button("Use example review", use_container_width=True)

    if use_example_clicked:
        text = example
        analyze_clicked = True

    if analyze_clicked:
        if not text or not text.strip():
            st.error("Please enter a review before clicking Analyze.")
            return

        try:
            with st.spinner("Loading models (first run only) ..."):
                warm_up_models()
        except ModelsNotTrainedError as exc:
            st.error(str(exc))
            st.info(
                "Train the models first:\n\n"
                "```\n"
                "python -m src.data.download_data\n"
                "python -m src.data.prepare_data\n"
                "python -m src.training.train_aspect\n"
                "python -m src.training.train_sentiment\n"
                "```"
            )
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"An unexpected error occurred while loading models: {exc}")
            return

        try:
            with st.spinner("Analyzing review ..."):
                results = predict_absa(text)
        except Exception as exc:  # noqa: BLE001
            st.error(f"An error occurred during inference: {exc}")
            return

        render_results(text, results)


if __name__ == "__main__":
    main()
