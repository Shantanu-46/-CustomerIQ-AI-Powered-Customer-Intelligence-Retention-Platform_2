"""Load trained artifacts and score customers end-to-end: Predict -> Segment -> Explain -> Recommend."""
import json

import joblib
import pandas as pd

from .churn import FEATURES
from .config import MODELS_DIR, PREDICTION_THRESHOLD
from .explain import shap_values, top_factors
from .features import add_features
from .recommend import recommend, risk_level, segment_label
from .segmentation import assign_segments


class CustomerIQ:
    def __init__(self, models_dir=MODELS_DIR):
        self.churn_model = joblib.load(models_dir / "churn_model.joblib")
        self.explainer = joblib.load(models_dir / "explainer.joblib")
        self.segmenter = joblib.load(models_dir / "segmenter.joblib")
        try:
            self.threshold = json.loads((models_dir / "threshold.json").read_text())["decision_threshold"]
        except FileNotFoundError:
            self.threshold = PREDICTION_THRESHOLD

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        feat = add_features(raw).reset_index(drop=True)
        proba = self.churn_model.predict_proba(feat[FEATURES])[:, 1]
        seg = assign_segments(self.segmenter, feat)
        X, sv = shap_values(self.explainer, feat)
        factors = top_factors(self.explainer, feat, sv, X)
        thr = seg.segment.map(self.segmenter["low_engagement_thresholds"]).values
        low_eng = (feat["engagement_score"].values <= thr)

        rows = []
        for i in range(len(feat)):
            risk = risk_level(proba[i])
            actions = recommend(risk, seg.segment.iloc[i], bool(low_eng[i]), [f["feature"] for f in factors[i]])
            rows.append({
                "churn_prediction": "YES" if proba[i] >= self.threshold else "NO",
                "churn_probability": round(float(proba[i]), 4),
                "risk_level": risk,
                "segment": seg.segment.iloc[i],
                "segment_label": segment_label(seg.segment.iloc[i], bool(low_eng[i])),
                "top_factors": " | ".join(f["text"] for f in factors[i]),
                "recommended_actions": " | ".join(actions),
                "pca_1": round(float(seg.pca_1.iloc[i]), 4), "pca_2": round(float(seg.pca_2.iloc[i]), 4),
            })
        out = pd.DataFrame(rows)
        if "customer_id" in raw.columns:
            out.insert(0, "customer_id", raw["customer_id"].values)
        return out
