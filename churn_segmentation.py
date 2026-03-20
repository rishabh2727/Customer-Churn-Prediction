import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _pick_k_silhouette(
    X_scaled: np.ndarray,
    candidate_ks: List[int],
    random_state: int,
    sample_size: int = 2500,
) -> Tuple[int, Dict[int, float]]:
    """
    Pick K by maximizing silhouette score on a random sample.

    silhouette_score can be expensive on full datasets (pairwise distances),
    so we compute it on a sample to keep runtime reasonable.
    """
    rng = np.random.default_rng(random_state)
    n = X_scaled.shape[0]
    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)

    X_sample = X_scaled[idx]
    scores: Dict[int, float] = {}
    best_k = candidate_ks[0]
    best_score = -1.0

    for k in candidate_ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        scores[k] = float(score)
        if score > best_score:
            best_k = k
            best_score = float(score)

    return best_k, scores


def _label_cluster(
    *,
    mean_risk: float,
    mean_balance: float,
    mean_tenure: float,
    mean_active: float,
    mean_satisfaction: float,
    mean_complain: float,
    thresholds: Dict[str, float],
) -> str:
    """
    Heuristic rules to create bank-friendly segment names.
    """
    risk_high = mean_risk >= thresholds["risk_high"]
    risk_low = mean_risk < thresholds["risk_high"]

    value_high = mean_balance >= thresholds["value_high"]
    value_low = mean_balance < thresholds["value_high"]

    tenure_loyal = mean_tenure >= thresholds["tenure_loyal"]
    tenure_not_loyal = mean_tenure < thresholds["tenure_loyal"]

    active_high = mean_active >= thresholds["active_high"]
    active_low = mean_active < thresholds["active_high"]

    satisfaction_high = mean_satisfaction >= thresholds["satisfaction_high"]
    satisfaction_low = mean_satisfaction < thresholds["satisfaction_high"]

    complain_high = mean_complain >= thresholds["complain_high"]

    # At-risk: high churn risk + low engagement/loyalty signals.
    if risk_high and tenure_not_loyal and active_low:
        return "at risk"

    # Loyal high value: low risk + strong value + (long tenure OR strong engagement).
    if risk_low and value_high and (tenure_loyal or active_high):
        return "loyal high value"

    # Low engagement: low risk and weaker engagement/experience.
    # (complaints are a weak proxy for frustration; keep as a soft signal)
    if risk_low and value_low and active_low and satisfaction_low:
        return "low engagement"

    # Complaints-heavy but not necessarily highest risk.
    if complain_high and risk_high:
        return "at risk (complaints)"

    return "other"


def segment_customers(
    data_path: str,
    scores_path: str,
    out_segments_path: str,
    out_summary_path: str,
    random_state: int = 42,
) -> None:
    df = pd.read_csv(data_path)
    scores = pd.read_csv(scores_path)

    # Merge churn risk features into the original customer table.
    merged = df.merge(scores, on="CustomerId", how="inner")

    # Use an ensemble probability for clustering stability.
    # Columns were created by churn_modeling.py.
    proba_cols = [
        c
        for c in merged.columns
        if c in ("logistic_regression_churn_proba", "xgboost_churn_proba", "neural_net_mlp_churn_proba")
    ]
    if len(proba_cols) < 1:
        raise ValueError(f"Could not find churn proba columns in `{scores_path}`. Found: {list(merged.columns)}")

    merged["churn_proba_avg"] = merged[proba_cols].mean(axis=1)

    # Engagement/value feature set for KMeans (numeric + binary only).
    # Important: we DO NOT include churn risk inside the KMeans features.
    # Clustering should be unsupervised in the customer-behavior space;
    # we add churn-based cluster labels after clustering.
    cluster_features = [
        "Balance",
        "EstimatedSalary",
        "Tenure",
        "Age",
        "CreditScore",
        "NumOfProducts",
        "IsActiveMember",
        "Complain",
        "Satisfaction Score",
        "HasCrCard",
        "Point Earned",
    ]
    missing = [c for c in cluster_features if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing expected features in merged dataset: {missing}")

    X = merged[cluster_features].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    candidate_ks = [3, 4, 5, 6]
    k, silhouette_by_k = _pick_k_silhouette(
        X_scaled=X_scaled,
        candidate_ks=candidate_ks,
        random_state=random_state,
        sample_size=2500,
    )

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    merged["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Thresholds for naming clusters (quantile-based, robust to scaling).
    thresholds = {
        # churn_proba_avg is highly bimodal in this dataset (near ~0.001 vs ~0.99),
        # so we use a high quantile to reserve "at risk" for the extreme cluster.
        "risk_high": float(merged["churn_proba_avg"].quantile(0.8)),
        "value_high": float(merged["Balance"].quantile(0.66)),
        # Tenure is in 0..10 with many values around 5, so use the median.
        "tenure_loyal": float(merged["Tenure"].quantile(0.5)),
        # For a binary feature, using a quantile often yields 0 or 1.
        # Using 0.55 separates "mostly inactive" clusters from the rest.
        "active_high": 0.55,
        "satisfaction_high": float(merged["Satisfaction Score"].quantile(0.66)),
        "complain_high": 1.0,
    }

    cluster_rows: List[Dict[str, object]] = []
    segment_names_by_cluster: Dict[int, str] = {}

    for cluster_id, grp in merged.groupby("cluster_id"):
        mean_risk = _safe_mean(grp["churn_proba_avg"])
        mean_balance = _safe_mean(grp["Balance"])
        mean_tenure = _safe_mean(grp["Tenure"])
        mean_active = _safe_mean(grp["IsActiveMember"])
        mean_satisfaction = _safe_mean(grp["Satisfaction Score"])
        mean_complain = _safe_mean(grp["Complain"])

        segment_name = _label_cluster(
            mean_risk=mean_risk,
            mean_balance=mean_balance,
            mean_tenure=mean_tenure,
            mean_active=mean_active,
            mean_satisfaction=mean_satisfaction,
            mean_complain=mean_complain,
            thresholds=thresholds,
        )
        segment_names_by_cluster[int(cluster_id)] = segment_name

        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "segment_name": segment_name,
                "count": int(len(grp)),
                "mean_churn_proba_avg": mean_risk,
                "mean_balance": mean_balance,
                "mean_tenure": mean_tenure,
                "mean_is_active_member": mean_active,
                "mean_satisfaction_score": mean_satisfaction,
                "mean_complain": mean_complain,
            }
        )

    summary = {
        "data_path": data_path,
        "scores_path": scores_path,
        "selected_k": k,
        "silhouette_by_k": silhouette_by_k,
        "n_customers": int(len(merged)),
        "cluster_feature_names": cluster_features,
        "thresholds_used_for_naming": thresholds,
        "clusters": sorted(cluster_rows, key=lambda r: r["count"], reverse=True),
    }

    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Pick a single risk score to show alongside segments.
    # Prefer logistic regression if available.
    risk_score_col = (
        "logistic_regression_churn_risk_0_100"
        if "logistic_regression_churn_risk_0_100" in merged.columns
        else proba_cols[0].replace("_churn_proba", "_churn_risk_0_100")
    )
    if risk_score_col not in merged.columns:
        # Fallback: no risk score available; still output proba avg.
        merged["risk_score_0_100"] = np.rint(merged["churn_proba_avg"].clip(0, 1) * 100).astype(int)
        risk_score_col = "risk_score_0_100"

    out = merged[["CustomerId", "cluster_id", risk_score_col, "churn_proba_avg"]].copy()
    out["segment_name"] = out["cluster_id"].map(segment_names_by_cluster)
    out = out.sort_values(["segment_name", risk_score_col], ascending=[True, False])
    out.to_csv(out_segments_path, index=False)

    # Print a compact view for you.
    print(f"Segmentation produced {k} clusters.")
    top_clusters = sorted(cluster_rows, key=lambda r: r["count"], reverse=True)
    for r in top_clusters:
        print(
            f"cluster {r['cluster_id']}: {r['segment_name']} | count={r['count']} "
            f"| mean_risk={r['mean_churn_proba_avg']:.4f} | mean_balance={r['mean_balance']:.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="KMeans customer segmentation (bank-friendly labels).")
    parser.add_argument("--data", default="Customer-Churn-Records.csv")
    parser.add_argument("--scores", default="churn_risk_scores.csv")
    parser.add_argument("--out", default="churn_segments.csv")
    parser.add_argument("--summary", default="segmentation_summary.json")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    segment_customers(
        data_path=args.data,
        scores_path=args.scores,
        out_segments_path=args.out,
        out_summary_path=args.summary,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

