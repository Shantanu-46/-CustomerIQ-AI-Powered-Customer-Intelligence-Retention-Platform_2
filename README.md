# CustomerIQ — AI-Powered Customer Intelligence & Retention Platform

An industry-style machine-learning project that analyses customer behaviour, **predicts churn**,
**segments customers**, **explains** the drivers with SHAP, and **recommends** retention actions.

```
Predict  →  Segment  →  Explain  →  Recommend
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py            # generate data, train, evaluate, explain, segment, score (~1 min)
streamlit run app/dashboard.py    # interactive dashboard  -> http://localhost:8501
uvicorn api.main:app --reload     # REST API + Swagger     -> http://127.0.0.1:8000/docs
pytest -q                         # tests
```
Trained models, scored data and reports are already included, so the dashboard and API work immediately.
Use `python run_pipeline.py --regenerate` to rebuild everything, or `--data your.csv` to use your own data
(same columns as `data/raw/customers.csv`).

## What's inside

| Layer | Details |
|---|---|
| **Supervised** | Logistic Regression, Decision Tree, Random Forest, KNN, SVM (+ XGBoost). Class imbalance handled via class weights; compared on ROC-AUC / PR-AUC / F1. Best model is **probability-calibrated** and output as churn prediction + probability + risk level (Low <30%, Medium 30–60%, High ≥60%). |
| **Unsupervised** | K-Means (k chosen with elbow + silhouette), Hierarchical clustering (cross-check via ARI), PCA for the 2D cluster plot. Clusters are auto-named: High-Value, Regular, Low-Engagement, At-Risk. |
| **Explainable AI** | SHAP `TreeExplainer` on XGBoost → global feature importance + top churn drivers per customer. |
| **Recommendations** | Rule engine combining risk, segment and SHAP drivers → e.g. *Contact customer → Offer retention campaign → Investigate support issues → Increase product engagement*. |
| **Serving** | FastAPI (`/predict`, `/customers/{id}`, `/summary`) and a 4-page Streamlit dashboard. |

## Project layout

```
customeriq/
├── run_pipeline.py            # end-to-end training + scoring
├── requirements.txt
├── src/customeriq/
│   ├── config.py              # paths, thresholds, constants
│   ├── data.py                # synthetic data generator (25,430 customers)
│   ├── features.py            # feature engineering + preprocessing
│   ├── churn.py               # supervised models, evaluation, calibration
│   ├── segmentation.py        # K-Means / hierarchical / PCA
│   ├── explain.py             # SHAP importance + per-customer drivers
│   ├── recommend.py           # risk bands + recommendation rules
│   └── inference.py           # CustomerIQ class: raw data -> full output
├── api/main.py                # FastAPI service
├── app/dashboard.py           # Streamlit dashboard
├── notebooks/CustomerIQ_walkthrough.ipynb   # executed walkthrough (EDA -> recommendations)
├── data/{raw,processed}/      # customers.csv, customers_scored.csv
├── models/                    # churn_model, explainer, segmenter (.joblib)
├── reports/                   # metrics CSV/JSON + figures/
└── tests/test_smoke.py
```

## Results on the bundled synthetic data

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | **0.873** | 0.608 | 0.388 | 0.796 | 0.521 |
| XGBoost | 0.868 | 0.600 | 0.646 | 0.405 | 0.498 |
| Random Forest | 0.866 | 0.583 | 0.476 | 0.672 | 0.557 |
| Decision Tree | 0.853 | 0.532 | 0.360 | 0.778 | 0.492 |
| KNN | 0.848 | 0.545 | 0.648 | 0.342 | 0.448 |
| SVM (8k-row subsample) | 0.846 | 0.478 | 0.384 | 0.776 | 0.513 |

(Metrics at each model's default 0.5 cut-off on a 20% stratified hold-out. The deployed model uses a
calibrated probability with a decision threshold tuned for F1 ≈ 0.39, giving F1 ≈ 0.58.)

Dashboard KPIs: 25,430 customers · 1,447 high-risk · 12.3% predicted churn · 4 segments.
Top SHAP drivers: days since last purchase, contract type, engagement, discount dependence, support tickets.

> **Important honesty note:** the dataset is *synthetic*, generated from a logistic-style churn process, which is why
> Logistic Regression edges out the tree models. On real data expect different rankings, so re-run
> `run_pipeline.py --data your.csv` and let the comparison decide. Real data must include the same columns
> (`customer_id, age, region, tenure_months, contract_type, payment_method, monthly_spend, total_spend, num_orders,
> logins_per_month, avg_session_minutes, days_since_last_purchase, support_tickets, discount_usage_rate, churn`).

## API example

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"tenure_months": 6, "logins_per_month": 2, "support_tickets": 5, "days_since_last_purchase": 120}'
```
```json
{"churn_prediction": "YES", "churn_probability": 0.9, "risk_level": "High",
 "segment": "At-Risk", "top_churn_factors": ["..."], "recommended_actions": ["Contact customer", "..."]}
```

## Ideas to extend
Hyperparameter tuning (Optuna), cross-validation, SQL/SQLite ingestion, drift monitoring, Dockerfile, CI, uplift modelling for campaign targeting.
