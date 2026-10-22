"""Feature engineering + preprocessing."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERIC = ["age", "tenure_months", "monthly_spend", "total_spend", "num_orders",
                "logins_per_month", "avg_session_minutes", "days_since_last_purchase",
                "support_tickets", "discount_usage_rate"]
ENGINEERED = ["orders_per_month", "avg_order_value", "tickets_per_month", "engagement_score"]
NUMERIC = BASE_NUMERIC + ENGINEERED
CATEGORICAL = ["contract_type", "payment_method", "region"]
TARGET = "churn"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive behavioural features. Uses fixed constants only (no data-dependent leakage)."""
    d = df.copy()
    tenure = d["tenure_months"].clip(lower=1)
    d["orders_per_month"] = d["num_orders"] / tenure
    d["avg_order_value"] = d["total_spend"] / d["num_orders"].clip(lower=1)
    d["tickets_per_month"] = d["support_tickets"] / tenure.clip(upper=6)   # tickets = last 6 months
    d["engagement_score"] = (d["logins_per_month"] / 12 + d["avg_session_minutes"] / 15) / 2
    return d


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
