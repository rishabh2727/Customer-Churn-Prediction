import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_RISK_COL = "logistic_regression_churn_risk_0_100"


def _age_group(age: float) -> str:
    # Bank-friendly bins; tweak if you prefer different cutoffs.
    if age < 25:
        return "Under 25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


@st.cache_data(show_spinner=False)
def load_data(
    churn_records_path: str,
    segments_path: str,
):
    churn = pd.read_csv(churn_records_path)
    seg = pd.read_csv(segments_path)

    required_seg_cols = {"CustomerId", "segment_name"}
    missing_seg = required_seg_cols - set(seg.columns)
    if missing_seg:
        raise ValueError(f"`{segments_path}` is missing columns: {sorted(missing_seg)}")

    churn = churn.copy()
    seg = seg.copy()

    # Merge: segment_name + risk score
    merged = churn.merge(seg, on="CustomerId", how="inner")

    # Derive age group + product labels
    merged["AgeGroup"] = merged["Age"].apply(_age_group)
    merged["ProductLabel"] = merged["NumOfProducts"].map(
        {
            1: "Single product",
            2: "2 products",
            3: "3 products",
            4: "4 products",
        }
    ).fillna(merged["NumOfProducts"].astype(str) + " products")

    # Choose a risk column (prefer logistic regression)
    if DEFAULT_RISK_COL in merged.columns:
        merged["risk_score_0_100"] = merged[DEFAULT_RISK_COL]
    else:
        # Fallback: first model's risk col
        risk_cols = [c for c in merged.columns if c.endswith("_risk_0_100")]
        if not risk_cols:
            raise ValueError("No risk score column found in merged data.")
        merged["risk_score_0_100"] = merged[risk_cols[0]]

    return merged


def _risk_bins(df: pd.DataFrame, bin_size: int = 10) -> pd.DataFrame:
    bins = list(range(0, 100 + bin_size, bin_size))
    labels = [f"{b}-{b + bin_size - 1}" for b in bins[:-1]]
    df = df.copy()
    df["risk_bin"] = pd.cut(df["risk_score_0_100"], bins=bins, labels=labels, include_lowest=True)
    out = df.groupby(["segment_name", "risk_bin"], dropna=False).size().reset_index(name="count")
    # Convert bins order back to numeric-ish
    out["risk_bin"] = out["risk_bin"].astype(str)
    return out.sort_values(["segment_name", "risk_bin"])


def main() -> None:
    st.set_page_config(page_title="Retention Dashboard", layout="wide")
    st.title("Customer Retention Dashboard")
    st.caption("Churn risk by segment + targeting views for geography, age group, and product.")

    colA, colB, colC = st.columns(3)
    with colA:
        churn_records_path = st.text_input(
            "Customer churn dataset",
            value=os.path.join(".", "Customer-Churn-Records.csv"),
        )
    with colB:
        segments_path = st.text_input(
            "Segment + risk file",
            value=os.path.join(".", "churn_segments.csv"),
        )
    with colC:
        refresh = st.button("Reload data")

    if refresh:
        st.cache_data.clear()

    try:
        df = load_data(churn_records_path, segments_path)
    except Exception as e:
        st.error(str(e))
        st.info("Run the churn + segmentation pipeline first:")
        st.code(
            'python3 churn_modeling.py --data "Customer-Churn-Records.csv" --out "churn_risk_scores.csv" --report "model_comparison_report.json"\n'
            'python3 churn_segmentation.py --data "Customer-Churn-Records.csv" --scores "churn_risk_scores.csv" --out "churn_segments.csv" --summary "segmentation_summary.json"'
        )
        return

    # Filters
    segments: List[str] = sorted(df["segment_name"].dropna().unique().tolist())
    geos: List[str] = sorted(df["Geography"].dropna().unique().tolist())
    age_groups: List[str] = sorted(df["AgeGroup"].dropna().unique().tolist())
    products: List[str] = sorted(df["ProductLabel"].dropna().unique().tolist())

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Filters")
        chosen_segments = st.multiselect("Segment", segments, default=segments)
        chosen_geos = st.multiselect("Geography", geos, default=geos)
        chosen_age_groups = st.multiselect("Age group", age_groups, default=age_groups)
        chosen_products = st.multiselect("Product", products, default=products)

        risk_threshold = st.slider(
            "Minimum risk score (0-100)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    filtered = df[
        (df["segment_name"].isin(chosen_segments))
        & (df["Geography"].isin(chosen_geos))
        & (df["AgeGroup"].isin(chosen_age_groups))
        & (df["ProductLabel"].isin(chosen_products))
        & (df["risk_score_0_100"] >= risk_threshold)
    ].copy()

    # KPI bar
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Customers in view", f"{len(filtered):,}")
    with kpi2:
        st.metric("Avg churn risk", f"{filtered['risk_score_0_100'].mean():.1f}")
    with kpi3:
        st.metric("Churn rate (Exited=1)", f"{filtered['Exited'].mean()*100:.1f}%")

    # Risk by segment
    st.subheader("Churn risk by segment")
    seg_summary = (
        filtered.groupby("segment_name")
        .agg(
            customers=("CustomerId", "count"),
            avg_risk=("risk_score_0_100", "mean"),
            churn_rate=("Exited", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    # Geography heatmap-ish (bar)
    st.subheader("Average churn risk by geography (within filters)")
    geo_summary = (
        filtered.groupby("Geography")
        .agg(customers=("CustomerId", "count"), avg_risk=("risk_score_0_100", "mean"))
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
    st.bar_chart(geo_summary.set_index("Geography")["avg_risk"])

    # Age group distribution
    st.subheader("Average churn risk by age group (within filters)")
    age_summary = (
        filtered.groupby("AgeGroup")
        .agg(customers=("CustomerId", "count"), avg_risk=("risk_score_0_100", "mean"))
        .reset_index()
    )
    st.bar_chart(age_summary.set_index("AgeGroup")["avg_risk"])

    # Product distribution
    st.subheader("Average churn risk by product (within filters)")
    prod_summary = (
        filtered.groupby("ProductLabel")
        .agg(customers=("CustomerId", "count"), avg_risk=("risk_score_0_100", "mean"))
        .reset_index()
    )
    st.bar_chart(prod_summary.set_index("ProductLabel")["avg_risk"])

    # Risk distribution by segment (hist-like)
    st.subheader("Risk distribution (binned) by segment")
    bins_df = _risk_bins(filtered, bin_size=10)
    # Simple grouped bar using Altair built into Streamlit/Altair dependency.
    # We only plot counts; risk-bin ordering is handled by string labels.
    chart_data = bins_df.pivot_table(
        index="risk_bin", columns="segment_name", values="count", aggfunc="sum", fill_value=0
    )
    st.bar_chart(chart_data)

    # Targeting table (top accounts)
    st.subheader("Top accounts to retain (highest risk)")
    top_n = st.slider("How many customers to show", min_value=10, max_value=200, value=50, step=10)
    risk_col = "risk_score_0_100"

    top = (
        filtered[["CustomerId", "segment_name", "Geography", "Age", "AgeGroup", "ProductLabel", risk_col, "Exited"]]
        .sort_values(risk_col, ascending=False)
        .head(top_n)
    )
    st.dataframe(top, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

