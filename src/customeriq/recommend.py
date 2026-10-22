"""Turn predictions + segments + SHAP drivers into actionable retention recommendations."""
from .config import RISK_THRESHOLDS

_FEATURE_ACTIONS = {
    "support_tickets": "Investigate support issues",
    "tickets_per_month": "Investigate support issues",
    "logins_per_month": "Increase product engagement",
    "avg_session_minutes": "Increase product engagement",
    "engagement_score": "Increase product engagement",
    "orders_per_month": "Increase product engagement",
    "days_since_last_purchase": "Send win-back offer",
    "contract_type": "Offer contract-upgrade / loyalty incentive",
    "tenure_months": "Run early-life onboarding programme",
    "discount_usage_rate": "Review pricing & discount strategy",
    "monthly_spend": "Offer tailored bundle / plan review",
}


def risk_level(p: float) -> str:
    lo, hi = RISK_THRESHOLDS
    return "Low" if p < lo else ("Medium" if p < hi else "High")


def recommend(risk: str, segment: str, low_engagement: bool, factor_features: list[str]) -> list[str]:
    if risk == "Low":
        return ["No action needed - keep monitoring"]
    actions = ["Contact customer"] if risk == "High" else ["Send proactive check-in"]
    actions.append("Offer retention campaign")
    if segment == "High-Value" and risk == "High":
        actions.append("Assign dedicated account manager")
    for f in factor_features:
        a = _FEATURE_ACTIONS.get(f)
        if a and a not in actions:
            actions.append(a)
    if low_engagement and "Increase product engagement" not in actions:
        actions.append("Increase product engagement")
    return actions


def segment_label(segment: str, low_engagement: bool) -> str:
    return f"{segment} / Low-Engagement" if low_engagement and segment in ("High-Value", "Regular") else segment
