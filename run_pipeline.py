"""End-to-end CustomerIQ training pipeline.

    python run_pipeline.py                 # generate data (if missing) + train + score
    python run_pipeline.py --regenerate    # rebuild the synthetic dataset first
    python run_pipeline.py --data my.csv   # use your own data (same columns as data/raw/customers.csv)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from customeriq import churn, explain, segmentation
from customeriq.config import (DATA_RAW, DATA_SCORED, FIGURES_DIR, MODELS_DIR, RANDOM_STATE,
                               REPORTS_DIR, TEST_SIZE)
from customeriq.data import load_or_generate
from customeriq.features import TARGET, add_features
from customeriq.inference import CustomerIQ


def main(args):
    print("[1/7] Loading data")
    df = pd.read_csv(args.data) if args.data else load_or_generate(DATA_RAW, args.regenerate)
    feat = add_features(df)
    print(f"      {len(df):,} customers | churn rate {df[TARGET].mean():.1%}")

    print("[2/7] Train/test split (stratified)")
    X, y = feat[churn.FEATURES], feat[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    print("[3/7] Supervised learning - comparing models")
    results, fitted = churn.evaluate_models(X_tr, y_tr, X_te, y_te)
    results.round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    churn.plot_roc(fitted, X_te, y_te)
    best = results.loc[0, "model"]
    print(f"      Best by ROC-AUC: {best}")

    print("[4/7] Calibrating best model")
    final = churn.fit_calibrated(best, X_tr, y_tr)
    proba_te = final.predict_proba(X_te)[:, 1]
    from sklearn.metrics import brier_score_loss, roc_auc_score
    calib = {"best_model": best, "roc_auc_test": float(roc_auc_score(y_te, proba_te)),
             "brier_score_test": float(brier_score_loss(y_te, proba_te)),
             "mean_predicted_prob_test": float(proba_te.mean()), "actual_churn_rate_test": float(y_te.mean())}
    joblib.dump(final, MODELS_DIR / "churn_model.joblib")

    # decision threshold = the probability cut-off that maximises F1 on the held-out set
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y_te, proba_te)
    f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    thr = float(th[int(np.argmax(f1s))])
    calib.update(decision_threshold=thr, f1_at_threshold=float(f1s.max()))
    (MODELS_DIR / "threshold.json").write_text(json.dumps({"decision_threshold": thr}))
    print(f"      decision threshold (max F1) = {thr:.3f} -> F1 {f1s.max():.3f}")

    print("[5/7] Explainable AI (SHAP on XGBoost)")
    exp_art = explain.build_explainer(fitted["XGBoost"], feat.loc[X_tr.index])
    joblib.dump(exp_art, MODELS_DIR / "explainer.joblib")
    _, sv_all = explain.shap_values(exp_art, feat)
    imp = explain.global_importance(exp_art, sv_all)
    imp.round(5).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    explain.plot_importance(imp)
    try:
        explain.plot_shap_summary(exp_art, feat)
    except Exception as e:                                  # plotting must never break training
        print("      (shap summary plot skipped:", e, ")")

    print("[6/7] Unsupervised learning - segmentation")
    proba_all = final.predict_proba(X)[:, 1]
    seg_art, diag = segmentation.fit_segmentation(feat, proba_all)
    joblib.dump(seg_art, MODELS_DIR / "segmenter.joblib")
    diag["k_selection"].round(4).to_csv(REPORTS_DIR / "k_selection.csv", index=False)
    segmentation.plot_k_selection(diag["k_selection"])
    segmentation.plot_clusters(segmentation.assign_segments(seg_art, feat))
    print(f"      silhouette(k=4)={diag['silhouette_k']:.3f} | hier-vs-kmeans ARI={diag['hierarchical_vs_kmeans_ARI']:.3f}")

    print("[7/7] Scoring all customers (Predict -> Segment -> Explain -> Recommend)")
    scored = CustomerIQ().score(df)
    out = df.reset_index(drop=True).merge(scored, on="customer_id")
    out.to_csv(DATA_SCORED, index=False)

    summary = {
        "total_customers": int(len(out)),
        "high_risk_customers": int((out.risk_level == "High").sum()),
        "predicted_churn_pct": round(float((out.churn_prediction == "YES").mean() * 100), 1),
        "customer_segments": int(out.segment.nunique()),
        "segment_counts": out.segment.value_counts().to_dict(),
        "calibration": calib, "segmentation": {k: v for k, v in diag.items() if k != "k_selection"},
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Model comparison ===\n", results.round(4).to_string(index=False))
    print("\n=== Dashboard KPIs ===")
    for k in ("total_customers", "high_risk_customers", "predicted_churn_pct", "customer_segments"):
        print(f"  {k}: {summary[k]:,}" if isinstance(summary[k], int) else f"  {k}: {summary[k]}")
    print("  segments:", summary["segment_counts"])
    print(f"\nDone. Scored data -> {DATA_SCORED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--data", default=None)
    main(ap.parse_args())
