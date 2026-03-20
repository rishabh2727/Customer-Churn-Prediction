import os
import json
import re
from typing import Dict, List, Optional, Tuple

def _load_env_file(path: str = ".env") -> None:
    """
    Minimal .env loader (key=value) to avoid relying on python-dotenv.

    - Only sets variables that are not already present in the environment.
    - Ignores blank lines and comments starting with '#'.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # If .env can't be read, we'll just rely on existing env vars.
        return


_load_env_file()



import numpy as np
import pandas as pd
import requests
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


def _tone_snippets(tone: str) -> Dict[str, str]:
    # Keep content bank-appropriate and non-technical.
    if tone == "supportive":
        return {
            "greeting": "Hi {name},",
            "closing": "Thanks for giving us the chance to help.\nCustomer Retention Team",
        }
    if tone == "premium":
        return {
            "greeting": "Hello {name},",
            "closing": "We appreciate your relationship.\nCustomer Retention Team",
        }
    return {
        "greeting": "Hi {name},",
        "closing": "If you'd like, reply to this email and we'll set it up for you.\nCustomer Retention Team",
    }


def _choose_primary_reason(row: pd.Series) -> Tuple[str, str]:
    """
    Returns (internal_driver, action_line).
    """
    # Note: in this dataset Complain and IsActiveMember are 0/1.
    complain = int(row.get("Complain", 0))
    satisfaction = float(row.get("Satisfaction Score", 0))
    is_active = int(row.get("IsActiveMember", 1))
    tenure = float(row.get("Tenure", 0))

    low_satisfaction = satisfaction <= 2.0
    short_tenure = tenure <= 3.0

    if complain == 1 or low_satisfaction:
        return (
            "experience_support",
            "We noticed recent service friction. A specialist is ready to help resolve it quickly.",
        )
    if is_active == 0:
        return (
            "re_engagement",
            "It looks like you haven't been active recently. We'd like to bring you back with a tailored perk.",
        )
    if short_tenure:
        return (
            "onboarding_help",
            "Since you joined more recently, we want to make sure you're getting the most value right away.",
        )
    return (
        "retention_offer",
        "To keep things smooth, we prepared a retention offer tailored to your profile.",
    )


def _generate_retention_email(
    row: pd.Series,
    *,
    brand_name: str,
    tone: str,
) -> Tuple[str, str]:
    # Name hygiene: use Surname if present; otherwise fallback to CustomerId.
    name = str(row.get("Surname", "")).strip() or str(row.get("CustomerId", "Customer"))
    age = row.get("Age", np.nan)
    age_group = row.get("AgeGroup", "your area")
    product_label = str(row.get("ProductLabel", "our products"))
    tenure = float(row.get("Tenure", 0))
    balance = float(row.get("Balance", 0))
    complain = int(row.get("Complain", 0))
    satisfaction = float(row.get("Satisfaction Score", np.nan))
    is_active = int(row.get("IsActiveMember", 1))
    products = int(row.get("NumOfProducts", 0))

    risk_score = int(row.get("risk_score_0_100", 0))
    urgency = "urgent" if risk_score >= 90 else "important"

    snippets = _tone_snippets(tone)
    _, action_line = _choose_primary_reason(row)

    # Value framing (simple, non-sensitive).
    value_line = ""
    if balance >= 100000:
        value_line = "As a high-value customer, we're offering a priority retention benefit."
    elif balance <= 20000:
        value_line = "We'd like to improve value for your current balance with a focused perk."
    else:
        value_line = "We're offering a retention perk designed to match your banking needs."

    if products >= 3:
        product_line = f"We also included an option to optimize your {product_label} setup."
    else:
        product_line = f"We'll personalize next steps for your {product_label} usage."

    complain_line = ""
    if complain == 1:
        complain_line = "We apologize for any frustration this may have caused."
    elif not np.isnan(satisfaction) and satisfaction >= 4.0 and is_active == 1:
        complain_line = "Thanks for staying engaged."

    subject = f"{brand_name}: Quick retention support for you ({urgency})"
    greeting = snippets["greeting"].format(name=name)
    body = "\n".join(
        [
            greeting,
            "",
            f"We noticed your churn risk is high ({risk_score}/100). To help you stay with {brand_name}, we prepared a personalized retention message for you.",
            "",
            action_line,
            complain_line,
            value_line,
            product_line,
            "",
            "If you'd like, reply to this email and tell us what matters most (lower fees, faster support, or better value). We'll tailor the next step.",
            "",
            f"Profile snapshot: {age_group}, tenure {tenure:.0f} years.",
            "",
            snippets["closing"],
        ]
    )
    body = body.replace("\n\n\n", "\n\n")
    return subject, body


def _build_customer_profile_summary(row: pd.Series) -> str:
    """
    Create a short, non-sensitive customer profile summary for the AI prompt.

    We avoid using name/PII; we only summarize banking attributes from the dataset.
    """
    age = row.get("Age", np.nan)
    gender = row.get("Gender", "N/A")
    tenure = row.get("Tenure", 0)
    geography = row.get("Geography", "N/A")
    products = int(row.get("NumOfProducts", 0))
    balance = float(row.get("Balance", 0))
    is_active = int(row.get("IsActiveMember", 1))
    satisfaction = float(row.get("Satisfaction Score", np.nan))
    complain = int(row.get("Complain", 0))

    tenure_years = float(tenure) if not np.isnan(tenure) else 0.0
    active_str = "active" if is_active == 1 else "inactive recently"

    satisfaction_str = "N/A" if np.isnan(satisfaction) else f"{satisfaction:.1f}/5"
    complain_str = "yes" if complain == 1 else "no"

    return (
        f"Customer is {age:.0f} year old {gender}, from {geography}. "
        f"Has been with the bank for {tenure_years:.0f} years, holds {products} product(s), "
        f"and has a balance of ${balance:,.0f}. "
        f"Customer is {active_str}. "
        f"Satisfaction score is {satisfaction_str}. "
        f"Has previously complained: {complain_str}."
    )


def _extract_subject_body(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to parse model output into (subject, body).
    Supports either JSON or simple "Subject:" / "Body:" formatting.
    """
    cleaned = text.strip()

    # Try JSON first
    try:
        obj = json.loads(cleaned)
        subject = obj.get("subject") or obj.get("Subject")
        body = obj.get("body") or obj.get("Body")
        if isinstance(subject, str) and isinstance(body, str):
            return subject.strip(), body.strip()
    except Exception:
        pass

    # Fallback: regex extraction
    subject_match = re.search(r"Subject\s*:\s*(.+)", cleaned, flags=re.IGNORECASE)
    body_match = re.search(r"Body\s*:\s*([\s\S]+)$", cleaned, flags=re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else None
    body = body_match.group(1).strip() if body_match else None
    return subject, body


def _call_openai_chat(
    *,
    api_key: str,
    model: str,
    customer_profile_summary: str,
    brand_name: str,
    tone: str,
) -> Tuple[str, str]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "temperature": 0.4,
        "max_tokens": 450,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a bank relationship manager. "
                    "Write a short, warm, personalized retention email. "
                    "Do not include sensitive personal data. "
                    "Output must be valid JSON with exactly two keys: subject and body."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Tone: {tone}\nBrand: {brand_name}\n\n"
                    "Customer profile summary:\n"
                    f"{customer_profile_summary}\n\n"
                    "Task: Write a ready-to-send retention email that is empathetic, "
                    "references the customer's profile (tenure, activity, satisfaction, complaints, products), "
                    "and suggests a next step (e.g., call a relationship manager) without making unrealistic promises.\n\n"
                    "Return JSON only like: {\"subject\": \"...\", \"body\": \"...\"}"
                ),
            },
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    subject, body = _extract_subject_body(content)
    if not subject or not body:
        raise ValueError(f"Could not parse OpenAI response into subject/body. Raw output: {content[:500]}")
    return subject, body


def _call_anthropic_messages(
    *,
    api_key: str,
    model: str,
    customer_profile_summary: str,
    brand_name: str,
    tone: str,
) -> Tuple[str, str]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {
        "model": model,
        "max_tokens": 600,
        "temperature": 0.4,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a bank relationship manager. "
                    "Write a short, warm, personalized retention email. "
                    "Do not include sensitive personal data. "
                    "Output must be valid JSON with exactly two keys: subject and body.\n\n"
                    f"Tone: {tone}\nBrand: {brand_name}\n\n"
                    "Customer profile summary:\n"
                    f"{customer_profile_summary}\n\n"
                    "Task: Write a ready-to-send retention email that is empathetic, "
                    "references the customer's profile (tenure, activity, satisfaction, complaints, products), "
                    "and suggests a next step (e.g., call a relationship manager) without making unrealistic promises.\n\n"
                    "Return JSON only like: {\"subject\": \"...\", \"body\": \"...\"}"
                ),
            }
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Anthropics returns: {"content":[{"type":"text","text":"..."}], ...}
    content = data["content"][0]["text"]
    subject, body = _extract_subject_body(content)
    if not subject or not body:
        raise ValueError(f"Could not parse Anthropic response into subject/body. Raw output: {content[:500]}")
    return subject, body


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

    st.divider()
    st.subheader("Automated Retention Email Generator (AI)")

    # Default: generate for the at-risk segment(s).
    default_email_segments = [s for s in segments if s.lower().startswith("at risk")]
    if not default_email_segments and segments:
        default_email_segments = [segments[0]]

    st.caption("Select one at-risk customer, then generate a ready-to-send personalized retention email.")

    # Session cache to avoid repeated API calls.
    st.session_state.setdefault("email_cache", {})

    email_settings_col1, email_settings_col2 = st.columns([1, 2])
    with email_settings_col1:
        email_segments = st.multiselect(
            "Email segment(s)",
            options=segments,
            default=default_email_segments,
        )
        email_min_risk = st.slider("Min risk score", min_value=0, max_value=100, value=80, step=1)
        dropdown_max = st.slider("Max customers in dropdown", min_value=25, max_value=500, value=200, step=25)
    with email_settings_col2:
        brand_name = st.text_input("Brand name", value="Your Bank")
        tone = st.selectbox("Message tone", options=["actionable", "supportive", "premium"], index=0)

    eligible = filtered.copy()
    eligible = eligible[eligible["segment_name"].isin(email_segments)]
    eligible = eligible[eligible["risk_score_0_100"] >= email_min_risk]
    eligible = eligible.sort_values("risk_score_0_100", ascending=False)
    eligible_for_dropdown = eligible.head(dropdown_max)

    if eligible_for_dropdown.empty:
        st.warning("No customers match your email filters in the current dashboard view.")
        return

    eligible_for_dropdown = eligible_for_dropdown.copy()
    eligible_for_dropdown["customer_dropdown_label"] = eligible_for_dropdown.apply(
        lambda r: f"CustomerId {int(r['CustomerId'])} | risk {int(r['risk_score_0_100'])}",
        axis=1,
    )
    selected_idx = st.selectbox(
        "Choose an at-risk customer",
        options=eligible_for_dropdown.index.tolist(),
        format_func=lambda i: eligible_for_dropdown.loc[i, "customer_dropdown_label"],
    )
    selected_row = eligible_for_dropdown.loc[selected_idx]

    st.divider()
    st.subheader("Customer profile summary")
    profile_summary = _build_customer_profile_summary(selected_row)
    st.text_area("Profile summary", value=profile_summary, height=140)

    cache_key = f"{int(selected_row['CustomerId'])}|{brand_name}|{tone}"
    cached = st.session_state["email_cache"].get(cache_key)

    if st.button("Generate retention email for selected customer", type="primary"):
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
            anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620").strip()

            # Prefer the first configured credential set.
            if openai_api_key:
                subject, body = _call_openai_chat(
                    api_key=openai_api_key,
                    model=openai_model,
                    customer_profile_summary=profile_summary,
                    brand_name=brand_name,
                    tone=tone,
                )
            elif anthropic_api_key:
                subject, body = _call_anthropic_messages(
                    api_key=anthropic_api_key,
                    model=anthropic_model,
                    customer_profile_summary=profile_summary,
                    brand_name=brand_name,
                    tone=tone,
                )
            else:
                raise ValueError("Missing API credentials. Add the key(s) to the .env file in the repo root.")

            st.session_state["email_cache"][cache_key] = {"subject": subject, "body": body}
            cached = st.session_state["email_cache"][cache_key]
        except Exception as e:
            st.error(f"Email generation failed: {e}")
            return

    if cached:
        st.subheader("Ready-to-send email")
        st.text_input("Email subject", value=cached["subject"])
        st.text_area("Email body", value=cached["body"], height=260)


if __name__ == "__main__":
    main()

