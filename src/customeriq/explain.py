"""Explainable AI with SHAP (TreeExplainer on the XGBoost churn model)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import FIGURES_DIR, RANDOM_STATE
from .features import CATEGORICAL, NUMERIC

# (text when value is BELOW the population median, text when ABOVE)
PHRASES = {
    "logins_per_month": ("Low product usage", "Unusual login pattern"),
    "avg_session_minutes": ("Short session time", "Very long sessions"),
    "engagement_score": ("Low engagement", "High engagement"),
    "support_tickets": ("Few support tickets", "High support tickets"),
    "tickets_per_month": ("Low support-ticket rate", "High support-ticket rate"),
    "days_since_last_purchase": ("Recent purchase", "Long time since last purchase"),
    "tenure_months": ("Short tenure", "Long tenure"),
    "monthly_spend": ("Low monthly spend", "High monthly spend"),
    "total_spend": ("Low lifetime spend", "High lifetime spend"),
    "orders_per_month": ("Low purchase frequency", "High purchase frequency"),
    "num_orders": ("Few orders", "Many orders"),
    "avg_order_value": ("Low order value", "High order value"),
    "discount_usage_rate": ("Low discount usage", "Heavy discount dependence"),
    "age": ("Younger customer", "Older customer"),
}


def _raw_feature(transformed_name: str) -> str:
    """'num__age' -> 'age'; 'cat__contract_type_One year' -> 'contract_type'."""
    n = transformed_name.split("__", 1)[1]
    for c in CATEGORICAL:
        if n.startswith(c + "_"):
            return c
    return n


def build_explainer(pipe, feat_train: pd.DataFrame) -> dict:
    """pipe = fitted Pipeline(prep -> XGBClassifier)."""
    return {"pipe": pipe,
            "names": list(pipe.named_steps["prep"].get_feature_names_out()),
            "medians": feat_train[NUMERIC].median().to_dict()}


def shap_values(art: dict, feat: pd.DataFrame):
    X = art["pipe"].named_steps["prep"].transform(feat)
    sv = shap.TreeExplainer(art["pipe"].named_steps["model"]).shap_values(X)
    return X, np.asarray(sv)


def _factor_text(name: str, feat_row: pd.Series, medians: dict) -> tuple[str, str]:
    raw = _raw_feature(name)
    if raw in CATEGORICAL:
        cat = name.split("__", 1)[1][len(raw) + 1:]
        return raw, f"{raw.replace('_', ' ').title()}: {cat}"
    low, high = PHRASES[raw]
    return raw, low if feat_row[raw] <= medians[raw] else high


def top_factors(art: dict, feat: pd.DataFrame, sv: np.ndarray, X: np.ndarray, k: int = 3):
    """Per customer: the k features pushing churn risk UP the most (positive SHAP)."""
    names, med = art["names"], art["medians"]
    out = []
    for i in range(len(feat)):
        row, order, factors, seen = feat.iloc[i], np.argsort(-sv[i]), [], set()
        for j in order:
            if sv[i, j] <= 0 or len(factors) >= k:
                break
            n = names[j]
            if n.startswith("cat__") and X[i, j] == 0:      # category the customer does NOT have
                continue
            raw, text = _factor_text(n, row, med)
            if raw in seen:
                continue
            seen.add(raw)
            factors.append({"feature": raw, "text": text, "shap": float(sv[i, j])})
        out.append(factors)
    return out


def global_importance(art: dict, sv: np.ndarray) -> pd.DataFrame:
    """Mean |SHAP| aggregated from one-hot columns back to raw features."""
    df = pd.DataFrame({"feature": [_raw_feature(n) for n in art["names"]],
                       "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return (df.groupby("feature", as_index=False).sum()
              .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def plot_importance(imp: pd.DataFrame, path=FIGURES_DIR / "feature_importance.png", top=12):
    d = imp.head(top).iloc[::-1]
    plt.figure(figsize=(7.5, 5)); plt.barh(d.feature, d.mean_abs_shap, color="tab:blue")
    plt.xlabel("Mean |SHAP value|"); plt.title("Feature importance (SHAP)")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_shap_summary(art: dict, feat: pd.DataFrame, path=FIGURES_DIR / "shap_summary.png", n=2000):
    s = feat.sample(min(n, len(feat)), random_state=RANDOM_STATE)
    X, sv = shap_values(art, s)
    shap.summary_plot(sv, X, feature_names=art["names"], show=False, max_display=12)
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
