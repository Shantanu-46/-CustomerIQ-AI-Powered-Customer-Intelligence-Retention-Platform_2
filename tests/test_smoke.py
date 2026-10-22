import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from customeriq.data import generate_customers
from customeriq.features import add_features
from customeriq.recommend import recommend, risk_level


def test_data_shape_and_rate():
    df = generate_customers(n=5000)
    assert len(df) == 5000 and 0.10 < df.churn.mean() < 0.20


def test_features():
    f = add_features(generate_customers(n=200))
    assert {"orders_per_month", "engagement_score"} <= set(f.columns)


def test_risk_bands():
    assert (risk_level(0.1), risk_level(0.45), risk_level(0.874)) == ("Low", "Medium", "High")


def test_recommendations_high_risk():
    acts = recommend("High", "High-Value", True, ["support_tickets", "logins_per_month"])
    assert acts[:2] == ["Contact customer", "Offer retention campaign"]
    assert "Investigate support issues" in acts and "Increase product engagement" in acts


def test_end_to_end_api():
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/predict", json={}).json()
    assert r["risk_level"] in {"Low", "Medium", "High"} and r["recommended_actions"]
