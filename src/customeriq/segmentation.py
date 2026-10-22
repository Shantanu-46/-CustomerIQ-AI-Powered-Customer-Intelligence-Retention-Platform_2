"""Unsupervised learning: customer segmentation (K-Means, Hierarchical, PCA)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import FIGURES_DIR, N_CLUSTERS, RANDOM_STATE

SEGMENT_FEATURES = ["monthly_spend", "total_spend", "orders_per_month", "logins_per_month",
                    "avg_session_minutes", "days_since_last_purchase", "support_tickets"]
SEGMENT_NAMES = ["High-Value", "Regular", "Low-Engagement", "At-Risk"]


def _prep() -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("log", FunctionTransformer(np.log1p)),      # tame skewed money/recency
                     ("sc", StandardScaler())])


def select_k(X: np.ndarray, k_range=range(2, 9), sample=6000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X[idx], km.labels_[idx])})
    return pd.DataFrame(rows)


def plot_k_selection(ks: pd.DataFrame, path=FIGURES_DIR / "k_selection.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks.k, ks.inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
    ax[1].plot(ks.k, ks.silhouette, "o-", c="tab:orange"); ax[1].set(title="Silhouette score", xlabel="k")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def _name_clusters(feat: pd.DataFrame, labels: np.ndarray, churn_proba: np.ndarray) -> dict:
    """Give each cluster a business name from its profile (one-to-one assignment)."""
    prof = feat.assign(cluster=labels, p=churn_proba).groupby("cluster").agg(
        spend=("monthly_spend", "mean"), logins=("logins_per_month", "mean"), churn=("p", "mean"))
    left, names = set(prof.index), {}
    hv = prof.loc[list(left), "spend"].idxmax(); names[hv] = "High-Value"; left.remove(hv)
    ar = prof.loc[list(left), "churn"].idxmax(); names[ar] = "At-Risk"; left.remove(ar)
    le = prof.loc[list(left), "logins"].idxmin(); names[le] = "Low-Engagement"; left.remove(le)
    for c in left:
        names[c] = "Regular"
    return {int(k): v for k, v in names.items()}


def fit_segmentation(feat: pd.DataFrame, churn_proba: np.ndarray, k: int = N_CLUSTERS) -> tuple[dict, dict]:
    """Returns (artifacts, diagnostics). `feat` must contain engineered features."""
    prep = _prep()
    X = prep.fit_transform(feat[SEGMENT_FEATURES])
    ks = select_k(X)
    km = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)

    # hierarchical clustering on a sample as a cross-check of the K-Means structure
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    hier = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
    ari = adjusted_rand_score(km.labels_[idx], hier)

    names = _name_clusters(feat, km.labels_, churn_proba)
    seg_names = pd.Series([names[c] for c in km.labels_], index=feat.index)
    # "low engagement *for its segment*": bottom 20% of engagement within each segment
    thr = feat["engagement_score"].groupby(seg_names).quantile(0.20).to_dict()
    artifacts = {"prep": prep, "kmeans": km, "pca": pca, "names": names,
                 "low_engagement_thresholds": {k: float(v) for k, v in thr.items()}}
    diag = {"k_selection": ks, "silhouette_k": float(ks.loc[ks.k == k, "silhouette"].iloc[0]),
            "hierarchical_vs_kmeans_ARI": float(ari),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist()}
    return artifacts, diag


def assign_segments(art: dict, feat: pd.DataFrame) -> pd.DataFrame:
    X = art["prep"].transform(feat[SEGMENT_FEATURES])
    cl = art["kmeans"].predict(X)
    xy = art["pca"].transform(X)
    return pd.DataFrame({"cluster": cl, "segment": [art["names"][c] for c in cl],
                         "pca_1": xy[:, 0], "pca_2": xy[:, 1]}, index=feat.index)


def plot_clusters(seg: pd.DataFrame, path=FIGURES_DIR / "cluster_pca.png", sample=6000):
    s = seg.sample(min(sample, len(seg)), random_state=RANDOM_STATE)
    plt.figure(figsize=(7.5, 6))
    for name, g in s.groupby("segment"):
        plt.scatter(g.pca_1, g.pca_2, s=8, alpha=.55, label=name)
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.title("Customer segments (PCA + K-Means)")
    plt.legend(markerscale=2); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
