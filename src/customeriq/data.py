"""Synthetic customer data generator (25,430 customers, ~14.8% churn)."""
import numpy as np
import pandas as pd

from .config import DATA_RAW, N_CUSTOMERS, RANDOM_STATE, TARGET_CHURN_RATE

# latent behavioural profiles: spend, logins, recency(days), tickets, tenure, session minutes
_PROFILES = {
    0: dict(spend=180, logins=22, recency=12, tickets=1.0, tenure=40, session=28),  # high value
    1: dict(spend=75,  logins=12, recency=30, tickets=1.0, tenure=28, session=16),  # regular
    2: dict(spend=55,  logins=4,  recency=60, tickets=1.2, tenure=20, session=7),   # low engagement
    3: dict(spend=60,  logins=6,  recency=95, tickets=3.0, tenure=12, session=9),   # at risk
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_customers(n: int = N_CUSTOMERS, seed: int = RANDOM_STATE,
                       target_rate: float = TARGET_CHURN_RATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.choice(4, size=n, p=[0.18, 0.40, 0.24, 0.18])
    P = {k: np.array([_PROFILES[s][k] for s in latent]) for k in _PROFILES[0]}

    # 15% of high-value customers are disengaged (High-Value / Low-Engagement)
    disengaged = (latent == 0) & (rng.random(n) < 0.15)
    P["logins"] = np.where(disengaged, 5, P["logins"])
    P["session"] = np.where(disengaged, 8, P["session"])
    P["recency"] = np.where(disengaged, 70, P["recency"])

    tenure = np.clip(rng.gamma(4, P["tenure"] / 4), 1, 72).round().astype(int)
    monthly_spend = P["spend"] * rng.lognormal(0, 0.25, n)
    logins = rng.poisson(P["logins"])
    session = np.clip(rng.normal(P["session"], P["session"] * 0.25), 1, None)
    recency = np.clip(rng.gamma(2, P["recency"] / 2), 0, 365).round().astype(int)
    tickets = rng.poisson(P["tickets"])
    num_orders = np.clip(rng.poisson(tenure * (0.4 + logins * 0.08)), 1, None)
    total_spend = monthly_spend * tenure * rng.uniform(0.85, 1.05, n)
    age = np.clip(rng.normal(40, 12, n), 18, 80).round().astype(int)
    discount = rng.beta(2, 5, n)

    contract = np.where(
        latent == 0,
        rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.30, 0.40, 0.30]),
        rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.28, 0.17]),
    )
    payment = rng.choice(["Credit Card", "Debit Card", "UPI", "Net Banking"], n, p=[0.3, 0.25, 0.3, 0.15])
    region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.24, 0.22, 0.16, 0.26, 0.12])

    # churn generated from behaviour (so models + SHAP have real signal to find)
    z = (0.55 * (tickets - 1.5) / 1.5 - 0.60 * (logins - 12) / 8 + 0.70 * (recency - 45) / 40
         - 0.50 * (tenure - 25) / 15 - 0.30 * (session - 15) / 10 + 0.40 * (discount - 0.29) / 0.16
         + 0.90 * (contract == "Month-to-month") - 0.50 * (contract == "Two year")
         - 0.008 * (age - 40) + rng.normal(0, 0.35, n))
    lo, hi = -15.0, 5.0
    for _ in range(60):                      # calibrate intercept -> target churn rate
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(z + mid).mean() < target_rate else (lo, mid)
    churn = (rng.random(n) < _sigmoid(z + mid)).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{10000 + i}" for i in range(n)],
        "age": age, "region": region, "tenure_months": tenure,
        "contract_type": contract, "payment_method": payment,
        "monthly_spend": monthly_spend.round(2), "total_spend": total_spend.round(2),
        "num_orders": num_orders, "logins_per_month": logins,
        "avg_session_minutes": session.round(1), "days_since_last_purchase": recency,
        "support_tickets": tickets, "discount_usage_rate": discount.round(3),
        "churn": churn,
    })
    # sprinkle missing values so the imputation step is genuinely exercised
    for col in ("avg_session_minutes", "discount_usage_rate"):
        df.loc[rng.random(n) < 0.015, col] = np.nan
    return df


def load_or_generate(path=DATA_RAW, regenerate: bool = False) -> pd.DataFrame:
    if regenerate or not path.exists():
        df = generate_customers()
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)
