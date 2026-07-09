# 🔎 Customer Health Checker

A Streamlit app that uses a real trained **Random Forest** model to predict whether a customer is **Healthy**, **At Risk**, or **Churned** — and lets you adjust a customer's info live to see how the AI's verdict changes.

Pick a real customer from the dataset, tweak their age, spend, usage, or contact recency with plain sliders, and watch the model's prediction and confidence update instantly. No mock data, no dummy logic — every prediction comes from `random_forest_health_model.pkl` running live.

## Features

- **Live AI verdict** — Healthy / At Risk / Churned, with a real confidence score from the model (not a hardcoded number)
- **Plain-language controls** — every slider uses real units (age in years, spend in dollars, days since contact) instead of confusing abstract multipliers
- **Recommended action** — a simple rule-based suggestion (e.g. "Billing Grace Period Outreach") based on the customer's info
- **What the AI looks at most** — a breakdown of which signals the model relies on most heavily, in plain English
- **Grounded in real data** — customers are sampled directly from the training dataset, not synthetically generated

## How It Works

1. `Model Training/Model-Training.py` trains a `RandomForestClassifier` (scikit-learn) on `balanced_customer_health_dataset.csv` and saves it to `Model Training/random_forest_health_model.pkl`.
2. `Front End/App.py` loads that trained model and a sample of 140 real customers.
3. Editing a customer's info in the UI recomputes their engineered features (spend rate, lifecycle ratio, etc.) using the same formulas as the dataset, then calls `model.predict()` / `model.predict_proba()` live — every verdict shown is a genuine model output.

## Tech Stack

- [Streamlit](https://streamlit.io) — UI and app framework
- [scikit-learn](https://scikit-learn.org) — Random Forest model
- [pandas](https://pandas.pydata.org) — data handling
- [joblib](https://joblib.readthedocs.io) — model serialization

## Project Structure

```
├── README.md
├── .streamlit/config.toml                       # App theme
├── Model Training/
│   ├── Model-Training.py                        # Trains and saves the Random Forest model
│   └── random_forest_health_model.pkl           # Trained model (compressed)
└── Front End/
    ├── App.py                                   # Streamlit app (UI + live inference)
    ├── balanced_customer_health_dataset.csv     # Customer sample source
    └── requirements.txt                         # Python dependencies
```

`Front End/App.py` locates the model in the sibling `Model Training/` folder automatically (resolved relative to its own file location), so it works regardless of which directory you launch it from.

## Running Locally

```bash
cd "Front End"
pip install -r requirements.txt
streamlit run App.py
```

The app opens at `http://localhost:8501`.

## Live Demo

[Add your Streamlit Community Cloud link here once deployed]

## Notes

- The model's Healthy / At Risk / Churned classes were confirmed by cross-checking payment delay and support call averages against the training data (`0` = Healthy, `1` = At Risk, `2` = Churned).
- Categorical fields (Gender, Subscription Type, Contract Length) are stored pre-encoded in the CSV; the mapping used for display labels is assumed alphabetical (a standard `LabelEncoder` default) and only affects what's *shown*, not what the model predicts.
