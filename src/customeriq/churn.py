"""Supervised learning: churn prediction (LogReg, Tree, RF, KNN, SVM, XGBoost)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import FIGURES_DIR, RANDOM_STATE
from .features import CATEGORICAL, NUMERIC, build_preprocessor

FEATURES = NUMERIC + CATEGORICAL
SVM_SAMPLE = 8000   # RBF-SVM scales ~O(n^2): train on a stratified subsample


def get_models() -> dict:
    rs = RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                                class_weight="balanced", n_jobs=-1, random_state=rs),
        "KNN": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=rs),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=rs),
    }


def _scores(pipe, X):
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)


def evaluate_models(X_train, y_train, X_test, y_test):
    """Fit every model, return (metrics DataFrame, {name: fitted pipeline})."""
    rows, fitted = [], {}
    for name, est in get_models().items():
        Xt, yt = X_train, y_train
        if name == "SVM" and len(X_train) > SVM_SAMPLE:
            idx = yt.groupby(yt).sample(frac=SVM_SAMPLE / len(yt), random_state=RANDOM_STATE).index
            Xt, yt = X_train.loc[idx], y_train.loc[idx]
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)]).fit(Xt, yt)
        pred, score = pipe.predict(X_test), _scores(pipe, X_test)
        rows.append({"model": name,
                     "roc_auc": roc_auc_score(y_test, score),
                     "pr_auc": average_precision_score(y_test, score),
                     "accuracy": accuracy_score(y_test, pred),
                     "precision": precision_score(y_test, pred, zero_division=0),
                     "recall": recall_score(y_test, pred),
                     "f1": f1_score(y_test, pred)})
        fitted[name] = pipe
        print(f"  {name:<20} ROC-AUC={rows[-1]['roc_auc']:.4f}  F1={rows[-1]['f1']:.4f}")
    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return res, fitted


def fit_calibrated(best_name: str, X_train, y_train):
    """Refit the best model with probability calibration so risk bands are meaningful
    (class_weight='balanced' inflates raw probabilities)."""
    base = Pipeline([("prep", build_preprocessor()), ("model", get_models()[best_name])])
    return CalibratedClassifierCV(clone(base), method="sigmoid", cv=3).fit(X_train, y_train)


def plot_roc(fitted: dict, X_test, y_test, path=FIGURES_DIR / "roc_curves.png"):
    plt.figure(figsize=(7, 5.5))
    for name, pipe in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, _scores(pipe, X_test))
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, _scores(pipe, X_test)):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves - churn models"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()
