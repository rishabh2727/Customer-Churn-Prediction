# Customer Churn Prediction (Risk Scores 0–100)

This repo trains and compares three churn models using your dataset:

- Logistic Regression (with probability calibration)
- XGBoost (with probability calibration)
- Neural Net (MLPClassifier, with probability calibration)

It outputs a **risk score from 0 to 100** for each customer (higher = higher churn risk).

## Run

```bash
python3 churn_modeling.py --data "Customer-Churn-Records.csv" --out "churn_risk_scores.csv" --report "model_comparison_report.json"
```

## Outputs

1. `churn_risk_scores.csv` (all 10,000 customers)
   - `CustomerId`
   - `<model>_churn_proba` (P(Exited=1))
   - `<model>_churn_risk_0_100` (risk score 0–100)
   - `Exited_true` (ground truth from the dataset; useful for sanity checks)

2. `model_comparison_report.json`
   - Held-out test metrics comparing the three models (AUC, Avg Precision, Brier score, F1@0.5).

## Customer Segmentation (KMeans)

This creates unsupervised customer segments and then labels each cluster into bank-friendly groups using the churn risk scores.

```bash
python3 churn_segmentation.py --data "Customer-Churn-Records.csv" --scores "churn_risk_scores.csv" --out "churn_segments.csv" --summary "segmentation_summary.json"
```

Outputs:

- `churn_segments.csv`
  - `CustomerId`
  - `cluster_id`
  - `logistic_regression_churn_risk_0_100`
  - `churn_proba_avg`
  - `segment_name` (e.g., `loyal high value`, `low engagement`, `at risk`)

- `segmentation_summary.json`
  - selected `K` by silhouette score and cluster-level averages used for naming.

