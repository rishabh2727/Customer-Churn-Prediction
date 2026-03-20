import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ModelArtifacts:
    name: str
    calibrated_model: object
    metrics: Dict[str, float]


def _risk_score_from_probability(p: np.ndarray) -> np.ndarray:
    """
    Convert churn probability to a 0-100 integer risk score.

    Note: probabilities are expected to be P(Exited=1). We clamp to [0, 1]
    to be safe with any calibration artifacts.
    """
    p = np.clip(p, 0.0, 1.0)
    return np.rint(p * 100.0).astype(int)


def _build_preprocessors(
    df: pd.DataFrame, target_col: str
) -> Tuple[List[str], List[str], ColumnTransformer, ColumnTransformer, ColumnTransformer]:
    feature_cols = [c for c in df.columns if c not in [target_col, "RowNumber", "CustomerId", "Surname"]]
    cat_features = [c for c in feature_cols if df[c].dtype == "object"]
    num_features = [c for c in feature_cols if c not in cat_features]

    # Keep numeric features compatible with sparse concatenation.
    preprocess_sparse = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Tree models can usually work without scaling, but scaling numeric features
    # also keeps everything consistent and doesn't harm much.
    preprocess_xgb = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # MLP needs dense input (and benefits from centering/scaling).
    preprocess_dense_mlp = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return num_features, cat_features, preprocess_sparse, preprocess_xgb, preprocess_dense_mlp


def _evaluate_threshold_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "precision_at_threshold": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_threshold": recall_score(y_true, y_pred, zero_division=0),
        "f1_at_threshold": f1_score(y_true, y_pred, zero_division=0),
    }


def train_and_compare(
    data_path: str,
    output_scores_path: str,
    output_report_path: str,
    random_state: int = 42,
) -> None:
    df = pd.read_csv(data_path)
    target_col = "Exited"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column `{target_col}`, but columns are: {list(df.columns)}")

    # Identify preprocessing
    num_features, cat_features, preprocess_sparse, preprocess_xgb, preprocess_dense_mlp = _build_preprocessors(
        df, target_col
    )

    feature_cols = [c for c in df.columns if c not in [target_col, "RowNumber", "CustomerId", "Surname"]]
    X = df[feature_cols]
    y = df[target_col].astype(int).values
    customer_ids = df["CustomerId"].values

    churn_rate = float(y.mean())
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols. Churn rate (Exited=1): {churn_rate:.4f}")
    print(f"Features: {len(num_features)} numeric, {len(cat_features)} categorical")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    # Use a small CV for calibration (fast enough; improves risk calibration).
    # We keep this deterministic with a fixed random_state.
    calib_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    models_to_run: List[Tuple[str, Pipeline]] = []

    # 1) Logistic Regression
    lr_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess_sparse),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    models_to_run.append(("logistic_regression", lr_pipe))

    # 2) XGBoost
    xgb_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess_xgb),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    gamma=0.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=max(1, int(np.floor(np.log2(os.cpu_count() or 2)))),
                    random_state=random_state,
                ),
            ),
        ]
    )
    models_to_run.append(("xgboost", xgb_pipe))

    # 3) Neural Net (MLP)
    mlp_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess_dense_mlp),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate="adaptive",
                    max_iter=400,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )
    models_to_run.append(("neural_net_mlp", mlp_pipe))

    results: List[ModelArtifacts] = []
    proba_by_model: Dict[str, np.ndarray] = {}

    for name, base_pipe in models_to_run:
        print(f"\nTraining {name} (calibrating probabilities)...")
        calibrated = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=calib_cv)
        # Calibration can produce noisy RuntimeWarnings on extreme but still valid scores.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            calibrated.fit(X_train, y_train)

        y_proba_test = calibrated.predict_proba(X_test)[:, 1]
        proba_by_model[name] = y_proba_test

        metrics = _evaluate_threshold_metrics(y_test, y_proba_test, threshold=0.5)
        print(
            f"{name}: AUC={metrics['roc_auc']:.4f}, AP={metrics['avg_precision']:.4f}, "
            f"F1@0.5={metrics['f1_at_threshold']:.4f}"
        )

        results.append(ModelArtifacts(name=name, calibrated_model=calibrated, metrics=metrics))

    # Refit calibrated models on the full dataset for production-style scoring.
    proba_all_by_model: Dict[str, np.ndarray] = {}
    for name, base_pipe in models_to_run:
        print(f"Scoring all customers with {name} (calibrated)...")
        calibrated_all = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=calib_cv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            calibrated_all.fit(X, y)
        proba_all_by_model[name] = calibrated_all.predict_proba(X)[:, 1]

    # Build risk scores for all customers, scaled to 0-100.
    out = pd.DataFrame({"CustomerId": customer_ids})
    for name in proba_all_by_model:
        out[f"{name}_churn_proba"] = proba_all_by_model[name]
        out[f"{name}_churn_risk_0_100"] = _risk_score_from_probability(proba_all_by_model[name])

    # Include ground-truth for reference (if you want to sanity-check in hindsight).
    out["Exited_true"] = y

    out.to_csv(output_scores_path, index=False)

    # Save a JSON report.
    report = {
        "data_path": data_path,
        "rows": int(df.shape[0]),
        "churn_rate": churn_rate,
        "random_state": random_state,
        "test_size": 0.2,
        "num_features": len(num_features),
        "cat_features": len(cat_features),
        "metrics_by_model": {r.name: r.metrics for r in results},
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print a quick top-N risk table for convenience.
    first_model = results[0].name if results else None
    if first_model:
        sort_col = f"{first_model}_churn_risk_0_100"
        top = out.sort_values(sort_col, ascending=False).head(10)[
            ["CustomerId", "Exited_true", sort_col]
        ]
        print("\nTop 10 risk customers (by first model's risk score):")
        print(top.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn models and output 0-100 risk scores.")
    parser.add_argument("--data", default="Customer-Churn-Records.csv")
    parser.add_argument("--out", default="churn_risk_scores.csv")
    parser.add_argument("--report", default="model_comparison_report.json")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_and_compare(
        data_path=args.data,
        output_scores_path=args.out,
        output_report_path=args.report,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

