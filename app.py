"""
app.py
======
Streamlit front-end for the Credit Risk Predictor.

A loan officer enters an applicant's details, and the app returns:
  * the model's estimated probability of rejection,
  * an APPROVE / DECLINE decision using the cost-optimised threshold, and
  * a plain-English explanation of which factors drove the decision (exact for
    the linear model; SHAP-based for tree models when `shap` is installed).

Run with:
    streamlit run app.py

It loads the artifacts produced by train_model.py (models/). Train first if
they are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models")


@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "credit_risk_model.joblib"
    # Safety net for fresh deployments: if the model artifact is missing, train
    # it once from the bundled CSV (no database needed). Cached, so this runs at
    # most once per container.
    if not model_path.exists():
        from train_model import train
        train(source="csv", csv_path="loan_approval_dataset.csv",
              target="loan_status", positive_label="Rejected",
              drop=["loan_id", "Loan_ID", "id", "Sex", "Gender"])
    pipeline = joblib.load(model_path)
    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)
    background = pd.read_csv(MODELS_DIR / "background.csv")
    return pipeline, meta, background


# Friendlier labels than the raw column names.
DISPLAY_NAMES = {
    "no_of_dependents": "Number of dependents",
    "education": "Education",
    "self_employed": "Self-employed",
    "income_annum": "Annual income (₹)",
    "loan_amount": "Loan amount (₹)",
    "loan_term": "Loan term (years)",
    "cibil_score": "CIBIL score",
    "residential_assets_value": "Residential assets value (₹)",
    "commercial_assets_value": "Commercial assets value (₹)",
    "luxury_assets_value": "Luxury assets value (₹)",
    "bank_asset_value": "Bank asset value (₹)",
}

# Tooltip text shown via the "?" icon next to a field. Amounts are in Indian
# Rupees (INR); CIBIL is India's main consumer credit score.
HELP_TEXTS = {
    "no_of_dependents": "Number of people financially dependent on the applicant.",
    "education": "Highest education level: Graduate or Not Graduate.",
    "self_employed": "Whether the applicant is self-employed (Yes) or salaried (No).",
    "income_annum": "Applicant's total annual income, in INR.",
    "loan_amount": "Requested loan amount, in INR.",
    "loan_term": "Loan repayment term, in years.",
    "cibil_score": "Credit bureau score from 300 to 900. Higher means better "
                   "creditworthiness; scores below ~550 are commonly rejected.",
    "residential_assets_value": "Declared value of residential property, in INR.",
    "commercial_assets_value": "Declared value of commercial property, in INR.",
    "luxury_assets_value": "Declared value of luxury assets (cars, jewellery, "
                           "etc.), in INR.",
    "bank_asset_value": "Value of assets held with banks (deposits, etc.), in INR.",
}


def display_name(col: str) -> str:
    return DISPLAY_NAMES.get(col, col.replace("_", " "))


def build_inputs(meta: dict) -> pd.DataFrame:
    """Render input widgets from the saved feature metadata and return a
    one-row DataFrame in the original column order."""
    values = {}
    cols = st.columns(2)
    feats = meta["features"]
    for i, name in enumerate(feats["order"]):
        col = cols[i % 2]
        help_text = HELP_TEXTS.get(name)
        if name in feats["numeric"]:
            spec = feats["numeric"][name]
            if spec.get("integer"):
                # Whole-number stepper (e.g. Job is 0,1,2,3 - no decimals).
                values[name] = col.number_input(
                    display_name(name),
                    min_value=int(spec["min"]),
                    max_value=int(spec["max"]),
                    value=int(round(spec["median"])),
                    step=1,
                    format="%d",
                    help=help_text,
                )
            else:
                values[name] = col.number_input(
                    display_name(name),
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["median"]),
                    help=help_text,
                )
        else:
            options = feats["categorical"][name]
            values[name] = col.selectbox(display_name(name), options, help=help_text)
    return pd.DataFrame([values])[feats["order"]]


def _pretty_label(raw: str, meta: dict, x_row: pd.DataFrame) -> str:
    """Turn an encoded feature name (e.g. 'num__cibil_score',
    'cat__education_Graduate') into a human-readable label that includes the
    applicant's actual value."""
    body = raw.split("__", 1)[-1]
    feats = meta["features"]

    # Numeric feature: show the applicant's value.
    if body in feats["numeric"]:
        val = x_row.iloc[0][body]
        val = int(val) if float(val).is_integer() else round(float(val), 2)
        return f"{display_name(body)} = {val}"

    # Categorical one-hot: recover the column and its selected value.
    for col, options in feats["categorical"].items():
        for opt in options:
            if body == f"{col}_{opt}":
                return f"{display_name(col)} = {opt}"
    return body.replace("_", " ")


def local_contributions(pipeline, x_row: pd.DataFrame, meta: dict,
                        background: pd.DataFrame):
    """Return a DataFrame [label, contribution] explaining THIS applicant.

    For a linear model the contribution of each feature to the risk score
    (log-odds of default) is exactly coef * value - an exact, dependency-free
    explanation. For a tree model we use SHAP if available, else fall back to
    global feature importances (clearly flagged as approximate).
    """
    pre = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = pre.get_feature_names_out()
    x_t = pre.transform(x_row)
    x_t = (x_t.toarray() if hasattr(x_t, "toarray") else x_t)[0]

    approximate = False
    if hasattr(clf, "coef_"):                      # linear model: exact
        contrib = clf.coef_[0] * x_t
    else:                                          # tree model
        try:
            import shap
            X_bg = pre.transform(background)
            X_bg = X_bg.toarray() if hasattr(X_bg, "toarray") else X_bg
            sv = shap.TreeExplainer(clf).shap_values(
                x_t.reshape(1, -1)
            )
            if isinstance(sv, list):
                sv = sv[1]
            elif getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1]
            contrib = np.asarray(sv)[0]
        except Exception:
            contrib = clf.feature_importances_     # unsigned, global
            approximate = True

    df = pd.DataFrame({
        "label": [_pretty_label(n, meta, x_row) for n in names],
        "contribution": contrib,
    })
    # Drop inactive one-hot columns (zero contribution) for readability.
    df = df[df["contribution"].abs() > 1e-9].copy()
    return df, approximate


def main():
    st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳",
                       layout="wide")
    st.title("💳 Credit Risk Predictor")

    if not (MODELS_DIR / "credit_risk_model.joblib").exists():
        st.error("Model artifacts not found. Run `python train_model.py` first.")
        st.stop()

    pipeline, meta, background = load_artifacts()
    threshold = meta["threshold"]

    with st.sidebar:
        st.header("Model")
        st.markdown("**Algorithm**")
        st.markdown(
            f"<span style='font-size:1rem'>{meta['model']}</span>",
            unsafe_allow_html=True,
        )
        st.metric(
            "Test ROC-AUC", f"{meta['test_roc_auc']:.3f}",
            help="Probability the model ranks a random declined applicant as "
                 "riskier than a random approved one. 0.5 = coin flip, "
                 "1.0 = perfect. "
                 f"{meta['test_roc_auc']:.2f} means it gets that ordering right "
                 f"~{meta['test_roc_auc']*100:.0f}% of the time on unseen data.",
        )
        st.metric("Decision threshold", f"{threshold:.2f}")
        st.caption(
            f"Threshold is cost-optimised: wrongly approving an applicant who "
            f"should be declined is treated as {meta['cost_fn']:.0f}x as costly "
            f"as wrongly declining a sound applicant."
        )

    st.subheader("Applicant details")
    x_row = build_inputs(meta)

    if st.button("Assess risk", type="primary"):
        proba = float(pipeline.predict_proba(x_row)[:, 1][0])
        decision = "DECLINE" if proba >= threshold else "APPROVE"

        c1, c2 = st.columns(2)
        c1.metric("Probability of rejection", f"{proba:.1%}")
        if decision == "DECLINE":
            c2.error(f"Decision: {decision}")
        else:
            c2.success(f"Decision: {decision}")

        st.subheader("Why this decision?")
        try:
            df, approximate = local_contributions(pipeline, x_row, meta, background)

            up = df[df["contribution"] > 0].sort_values(
                "contribution", ascending=False).head(4)
            down = df[df["contribution"] < 0].sort_values(
                "contribution").head(4)

            verb = "declined" if decision == "DECLINE" else "approved"
            st.markdown(
                f"This applicant was **{verb}** with an estimated **{proba:.1%}** "
                f"chance of rejection (decision cutoff {threshold:.0%})."
            )

            if not up.empty:
                st.markdown("**Factors that increased the assessed risk:**")
                st.markdown("\n".join(f"- {r}" for r in up["label"]))
            if not down.empty:
                st.markdown("**Factors that reduced the assessed risk:**")
                st.markdown("\n".join(f"- {r}" for r in down["label"]))

            st.markdown("**Contribution to the risk score**")
            chart = (
                pd.concat([up, down])
                .set_index("label")["contribution"]
                .sort_values()
            )
            st.bar_chart(chart)
            st.caption(
                "Bars to the right push the applicant toward rejection; "
                "bars to the left toward approval. "
                + ("(Approximate: global feature importances - install `shap` "
                   "for exact per-applicant values on this model.)"
                   if approximate else
                   "(Exact contributions to the model's log-odds.)")
            )
        except Exception as e:
            st.info(f"Explanation unavailable: {e}")


if __name__ == "__main__":
    main()
