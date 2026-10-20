"""Streamlit dashboard for the Intelligent Document Classification project."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "document_classifier.joblib"
DATABASE_PATH = BASE_DIR / "document_classification.db"

CATEGORIES = (
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.misc",
)

EXAMPLES = {
    "Baseball": "The team won the game after a dominant pitching performance.",
    "Computer graphics": "The graphics card renders high-resolution images and video.",
    "Medicine": "The doctor discussed treatment options for the patient's condition.",
    "Politics": "The government announced a policy following a political debate.",
}


def clean_text(text: str) -> str:
    """Apply the same basic cleaning used when the model was trained."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Loading the trained classifier...")
def load_model(model_path: str):
    """Load the saved pipeline once per Streamlit server session."""
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_evaluation_metrics(database_path: str) -> tuple[int | None, float | None, pd.DataFrame]:
    """Read the saved held-out-test results without mutating the database."""
    empty = pd.DataFrame(columns=["Category", "Test documents", "Correct", "Accuracy"])
    if not Path(database_path).exists():
        return None, None, empty

    try:
        with sqlite3.connect(database_path) as connection:
            summary = pd.read_sql_query(
                """
                SELECT COUNT(*) AS total_predictions, AVG(correct) AS accuracy
                FROM predictions
                """,
                connection,
            )
            by_class = pd.read_sql_query(
                """
                SELECT
                    actual_label AS Category,
                    COUNT(*) AS "Test documents",
                    SUM(correct) AS Correct,
                    AVG(correct) AS Accuracy
                FROM predictions
                GROUP BY actual_label
                ORDER BY actual_label
                """,
                connection,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return None, None, empty

    return (
        int(summary.loc[0, "total_predictions"]),
        float(summary.loc[0, "accuracy"]),
        by_class,
    )


def set_example() -> None:
    """Copy the selected example into the editable document field."""
    st.session_state.document_text = EXAMPLES[st.session_state.example_choice]


st.set_page_config(page_title="Document Classifier", page_icon="📄", layout="wide")
st.title("📄 Intelligent Document Classification")
st.caption("Four-topic classifier built with TF-IDF and a Linear SVM")

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run the notebook to create it, then reload this page.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:  # Show a useful application error instead of a blank page.
    st.error(f"The trained model could not be loaded: {error}")
    st.stop()

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

with st.sidebar:
    st.header("About the model")
    st.write("Trained on four 20 Newsgroups topics:")
    for category in CATEGORIES:
        st.caption(f"• {category}")

    total_predictions, accuracy, class_metrics = load_evaluation_metrics(str(DATABASE_PATH))
    st.divider()
    st.header("Held-out test results")
    if accuracy is None:
        st.info("Saved evaluation results are unavailable.")
    else:
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.caption(f"Based on {total_predictions:,} held-out test documents.")

st.subheader("Classify a document")
st.selectbox(
    "Load a sample document",
    options=list(EXAMPLES),
    index=None,
    placeholder="Choose an example",
    key="example_choice",
    on_change=set_example,
)

with st.form("classification_form"):
    document = st.text_area(
        "Paste document text",
        key="document_text",
        height=220,
        placeholder="Enter a paragraph about graphics, baseball, medicine, or politics...",
    )
    submitted = st.form_submit_button("Classify document", type="primary")

if submitted:
    cleaned_document = clean_text(document)
    if not cleaned_document:
        st.warning("Please enter some text containing letters.")
    else:
        prediction = model.predict([cleaned_document])[0]
        st.success(f"Predicted category: {prediction}")

        if hasattr(model, "decision_function"):
            scores = model.decision_function([cleaned_document])[0]
            score_table = pd.DataFrame(
                {"Category": model.classes_, "Relative SVM score": scores}
            ).sort_values("Relative SVM score", ascending=False, ignore_index=True)
            st.dataframe(
                score_table.style.format({"Relative SVM score": "{:.3f}"}),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "These are relative Linear SVM decision scores, not calibrated probabilities."
            )

if not class_metrics.empty:
    st.divider()
    st.subheader("Test performance by category")
    display_metrics = class_metrics.copy()
    display_metrics["Accuracy"] = display_metrics["Accuracy"].map("{:.1%}".format)
    st.dataframe(display_metrics, hide_index=True, width="stretch")
