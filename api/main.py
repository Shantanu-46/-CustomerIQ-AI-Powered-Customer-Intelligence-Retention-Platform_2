"""CustomerIQ REST API.   uvicorn api.main:app --reload   ->  http://127.0.0.1:8000/docs"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from customeriq.config import DATA_SCORED
from customeriq.inference import CustomerIQ

app = FastAPI(title="CustomerIQ API", version="1.0.0",
              description="Predict -> Segment -> Explain -> Recommend")
_engine = None
_scored = None


def engine() -> CustomerIQ:
    global _engine
    if _engine is None:
        _engine = CustomerIQ()
    return _engine


def scored() -> pd.DataFrame:
    global _scored
    if _scored is None:
        if not DATA_SCORED.exists():
            raise HTTPException(503, "Run `python run_pipeline.py` first.")
        _scored = pd.read_csv(DATA_SCORED).set_index("customer_id", drop=False)
    return _scored


class CustomerFeatures(BaseModel):
    customer_id: str = "NEW-CUSTOMER"
    age: int = Field(39, ge=18, le=100)
    region: str = "West"
    tenure_months: int = Field(9, ge=0)
    contract_type: str = "Month-to-month"
    payment_method: str = "UPI"
    monthly_spend: float = 69.5
    total_spend: float = 542.0
    num_orders: int = 3
    logins_per_month: int = 5
    avg_session_minutes: float = 11.5
    days_since_last_purchase: int = 53
    support_tickets: int = 4
    discount_usage_rate: float = 0.41


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/summary")
def summary():
    d = scored()
    return {"total_customers": len(d),
            "high_risk_customers": int((d.risk_level == "High").sum()),
            "predicted_churn_pct": round(float((d.churn_prediction == "YES").mean() * 100), 1),
            "customer_segments": int(d.segment.nunique()),
            "segment_counts": d.segment.value_counts().to_dict()}


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    d = scored()
    if customer_id not in d.index:
        raise HTTPException(404, f"{customer_id} not found")
    r = d.loc[customer_id]
    return {"customer_id": customer_id, "churn_prediction": r.churn_prediction,
            "churn_probability": r.churn_probability, "risk_level": r.risk_level,
            "segment": r.segment_label, "top_churn_factors": r.top_factors.split(" | ") if isinstance(r.top_factors, str) else [],
            "recommended_actions": r.recommended_actions.split(" | ")}


@app.post("/predict")
def predict(c: CustomerFeatures):
    out = engine().score(pd.DataFrame([c.model_dump()])).iloc[0]
    return {"customer_id": c.customer_id, "churn_prediction": out.churn_prediction,
            "churn_probability": out.churn_probability, "risk_level": out.risk_level,
            "segment": out.segment_label,
            "top_churn_factors": out.top_factors.split(" | ") if out.top_factors else [],
            "recommended_actions": out.recommended_actions.split(" | ")}
