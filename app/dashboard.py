"""CustomerIQ Streamlit dashboard.   streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from customeriq.config import DATA_SCORED, REPORTS_DIR

st.set_page_config(page_title="CustomerIQ", page_icon="📊", layout="wide")
SEG_ORDER = ["High-Value", "Regular", "Low-Engagement", "At-Risk"]
SEG_COLORS = {"High-Value": "#2E86AB", "Regular": "#7FB069", "Low-Engagement": "#F6AE2D", "At-Risk": "#D1495B"}
RISK_COLORS = {"Low": "#7FB069", "Medium": "#F6AE2D", "High": "#D1495B"}


@st.cache_data
def load():
    if not DATA_SCORED.exists():
        return None, None, None
    df = pd.read_csv(DATA_SCORED)
    imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    models = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    return df, imp, models


df, imp, models = load()
if df is None:
    st.error("No scored data found. Run `python run_pipeline.py` first.")
    st.stop()

st.sidebar.title("📊 CustomerIQ")
st.sidebar.caption("AI-Powered Customer Intelligence & Retention")
page = st.sidebar.radio("Navigate", ["Overview", "Segments", "Customer Lookup", "Model Performance"])

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Customer Intelligence Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("High-Risk Customers", f"{(df.risk_level == 'High').sum():,}")
    c3.metric("Predicted Churn", f"{(df.churn_prediction == 'YES').mean():.1%}")
    c4.metric("Customer Segments", df.segment.nunique())

    a, b = st.columns(2)
    risk = df.risk_level.value_counts().reindex(["Low", "Medium", "High"]).reset_index()
    risk.columns = ["risk_level", "customers"]
    a.plotly_chart(px.bar(risk, x="risk_level", y="customers", color="risk_level", color_discrete_map=RISK_COLORS,
                          title="Churn risk distribution"), width="stretch")
    seg = df.segment.value_counts().reindex(SEG_ORDER).reset_index()
    seg.columns = ["segment", "customers"]
    b.plotly_chart(px.pie(seg, names="segment", values="customers", color="segment", hole=.45,
                          color_discrete_map=SEG_COLORS, title="Customer segment distribution"), width="stretch")

    a, b = st.columns(2)
    d = df.assign(age_band=pd.cut(df.age, [17, 25, 35, 45, 55, 65, 100], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]),
                  tenure_band=pd.cut(df.tenure_months, [0, 6, 12, 24, 48, 100], labels=["0-6m", "7-12m", "13-24m", "25-48m", "48m+"]))
    by_age = d.groupby("age_band", observed=True).churn_probability.mean().reset_index()
    by_ten = d.groupby("tenure_band", observed=True).churn_probability.mean().reset_index()
    a.plotly_chart(px.bar(by_age, x="age_band", y="churn_probability", title="Avg churn probability by age",
                          labels={"churn_probability": "Avg churn prob."}), width="stretch")
    b.plotly_chart(px.bar(by_ten, x="tenure_band", y="churn_probability", title="Avg churn probability by tenure",
                          labels={"churn_probability": "Avg churn prob."}, color_discrete_sequence=["#F6AE2D"]), width="stretch")

    a, b = st.columns(2)
    rev = df.groupby("segment").monthly_spend.sum().reindex(SEG_ORDER).reset_index()
    a.plotly_chart(px.bar(rev, x="segment", y="monthly_spend", color="segment", color_discrete_map=SEG_COLORS,
                          title="Monthly revenue by segment", labels={"monthly_spend": "Monthly revenue"}), width="stretch")
    top = imp.head(10).iloc[::-1]
    b.plotly_chart(px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
                          title="Feature importance (mean |SHAP|)"), width="stretch")

    st.subheader("Customer behaviour patterns")
    cols = ["logins_per_month", "avg_session_minutes", "days_since_last_purchase", "support_tickets", "monthly_spend", "tenure_months"]
    prof = df.groupby("segment")[cols].mean()
    norm = (prof / prof.max()).reset_index().melt("segment", var_name="metric", value_name="relative level")
    st.plotly_chart(px.bar(norm, x="metric", y="relative level", color="segment", barmode="group",
                           color_discrete_map=SEG_COLORS, category_orders={"segment": SEG_ORDER}), width="stretch")

# ---------------------------------------------------------------- Segments
elif page == "Segments":
    st.title("Customer Segmentation (PCA + K-Means)")
    s = df.sample(min(8000, len(df)), random_state=1)
    st.plotly_chart(px.scatter(s, x="pca_1", y="pca_2", color="segment", color_discrete_map=SEG_COLORS,
                               opacity=.6, hover_data=["customer_id", "churn_probability"],
                               category_orders={"segment": SEG_ORDER}, height=560), width="stretch")
    st.subheader("Segment profiles")
    st.dataframe(df.groupby("segment").agg(customers=("customer_id", "count"), monthly_spend=("monthly_spend", "mean"),
                                           logins=("logins_per_month", "mean"), recency_days=("days_since_last_purchase", "mean"),
                                           tickets=("support_tickets", "mean"), churn_prob=("churn_probability", "mean"),
                                           high_risk=("risk_level", lambda x: (x == "High").sum())).reindex(SEG_ORDER).round(2),
                 width="stretch")

# ---------------------------------------------------------------- Lookup
elif page == "Customer Lookup":
    st.title("Customer 360° & Recommendation")
    default = "CUST-10245"
    hi = df[df.risk_level == "High"].sort_values("churn_probability", ascending=False).customer_id.head(20).tolist()
    cid = st.text_input("Customer ID", default).strip().upper()
    st.caption("Highest-risk examples: " + ", ".join(hi[:6]))
    r = df[df.customer_id == cid]
    if r.empty:
        st.warning("Customer not found.")
    else:
        r = r.iloc[0]
        a, b = st.columns([1, 1.4])
        fig = go.Figure(go.Indicator(mode="gauge+number", value=r.churn_probability * 100,
                                     number={"suffix": "%"}, title={"text": "Churn probability"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[r.risk_level]},
                                            "steps": [{"range": [0, 30], "color": "#e8f3e2"}, {"range": [30, 60], "color": "#fdf0d0"},
                                                      {"range": [60, 100], "color": "#f7d6db"}]}))
        fig.update_layout(height=280, margin=dict(t=60, b=10))
        a.plotly_chart(fig, width="stretch")
        b.markdown(f"### {r.customer_id}\n**Risk:** {r.risk_level}  \n**Churn prediction:** {r.churn_prediction}  \n"
                   f"**Segment:** {r.segment_label}")
        st.markdown("**Top churn factors (SHAP)**")
        for f in (r.top_factors.split(" | ") if isinstance(r.top_factors, str) else []):
            st.markdown(f"- {f}")
        st.markdown("**Recommended action**")
        for act in r.recommended_actions.split(" | "):
            st.markdown(f"→ {act}")
        st.markdown("**Predict → Segment → Explain → Recommend**")
        st.dataframe(r[["age", "region", "tenure_months", "contract_type", "monthly_spend", "num_orders", "logins_per_month",
                        "avg_session_minutes", "days_since_last_purchase", "support_tickets"]].astype(str).to_frame("value"))

# ---------------------------------------------------------------- Models
else:
    st.title("Model Performance")
    st.dataframe(models.style.format({c: "{:.3f}" for c in models.columns if c != "model"}).highlight_max(
        subset=[c for c in models.columns if c != "model"], color="#d8f0d0"), width="stretch")
    fig_dir = REPORTS_DIR / "figures"
    a, b = st.columns(2)
    if (fig_dir / "roc_curves.png").exists():
        a.image(str(fig_dir / "roc_curves.png"), caption="ROC curves")
    if (fig_dir / "shap_summary.png").exists():
        b.image(str(fig_dir / "shap_summary.png"), caption="SHAP summary (XGBoost)")
    if (REPORTS_DIR / "summary.json").exists():
        with st.expander("Run summary (JSON)"):
            st.json(json.loads((REPORTS_DIR / "summary.json").read_text()))
