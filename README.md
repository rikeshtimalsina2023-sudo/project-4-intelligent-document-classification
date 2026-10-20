# Project 4 — Intelligent Document Classification

An end-to-end NLP project that classifies documents into four 20 Newsgroups topics. It includes data cleaning, two model comparisons, evaluation, error analysis, a saved model, SQLite analytics, and a Streamlit dashboard.

## Results

The selected TF-IDF + Linear SVM model achieved **89.9% accuracy** on 1,434 held-out test documents.

| Category | Test accuracy |
| --- | ---: |
| `comp.graphics` | 92.4% |
| `rec.sport.baseball` | 92.8% |
| `sci.med` | 89.4% |
| `talk.politics.misc` | 83.7% |

## Repository contents

| Path | Purpose |
| --- | --- |
| `document_classification.ipynb` | Complete reproducible workflow: preparation, model comparison, evaluation, and persistence. |
| `app/dashboard.py` | Interactive Streamlit dashboard for live predictions and saved test metrics. |
| `models/document_classifier.joblib` | Saved TF-IDF + Linear SVM pipeline. |
| `data/processed/` | Cleaned train and held-out test data. |
| `document_classification.db` | SQLite database containing held-out prediction results. |
| `sql/analytics.sql` | Reusable SQL queries for the stored evaluation results. |

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard opens in your browser. Choose a sample or paste a document, then select **Classify document**. The displayed SVM decision scores are rankings, not probabilities.

## Reproduce the project

Open `document_classification.ipynb` from the repository root and run its cells in order. It fetches the dataset, trains the models, refreshes the processed CSV files, saves the selected model, and rebuilds the SQLite evaluation database.

## Verify the dashboard

```bash
python -m unittest discover -s tests -v
```
