"""
app.py
======
Streamlit front-end for the Credit Risk Predictor.

A loan officer enters an applicant's details, and the app returns:
  * the model's estimated probability of default,
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
        train(source="csv", csv_path="german_credit_data.csv",
              target="Risk", positive_label="bad")
    pipeline = joblib.load(model_path)
    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)
    background = pd.read_csv(MODELS_DIR / "background.csv")
    return pipeline, meta, background


def build_inputs(meta: dict) -> pd.DataFrame:
    """Render input widgets from the saved feature metadata and return a
    one-row DataFrame in the original column order."""
    values = {}
    cols = st.columns(2)
    feats = meta["features"]
    for i, name in enumerate(feats["order"]):
        col = cols[i % 2]
        if name in feats["numeric"]:
            spec = feats["numeric"][name]
            values[name] = col.number_input(
                name.replace("_", " "),
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec["median"]),
            )
        else:
            options = feats["categorical"][name]
            values[name] = col.selectbox(name.replace("_", " "), options)
    return pd.DataFrame([values])[feats["order"]]


def _pretty_label(raw: str, meta: dict, x_row: pd.DataFrame) -> str:
    """Turn an encoded feature name (e.g. 'num__Credit_amount',
    'cat__Purpose_car') into a human-readable label that includes the
    applicant's actual value."""
    body = raw.split("__", 1)[-1]
    feats = meta["features"]

    # Numeric feature: show the applicant's value.
    if body in feats["numeric"]:
        val = x_row.iloc[0][body]
        val = int(val) if float(val).is_integer() else round(float(val), 2)
        return f"{body.replace('_', ' ')} = {val}"

    # Categorical one-hot: recover the column and its selected value.
    for col, options in feats["categorical"].items():
        for opt in options:
            if body == f"{col}_{opt}":
                return f"{col.replace('_', ' ')} = {opt}"
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
            help="Probability the model ranks a random defaulter as riskier "
                 "than a random non-defaulter. 0.5 = coin flip, 1.0 = perfect. "
                 f"{meta['test_roc_auc']:.2f} means it gets that ordering right "
                 f"~{meta['test_roc_auc']*100:.0f}% of the time on unseen data.",
        )
        st.metric("Decision threshold", f"{threshold:.2f}")
        st.caption(
            f"Threshold is cost-optimised: a false negative (approving a "
            f"defaulter) is treated as {meta['cost_fn']:.0f}x as costly as a "
            f"false positive (declining a good applicant)."
        )

    st.subheader("Applicant details")
    x_row = build_inputs(meta)

    if st.button("Assess risk", type="primary"):
        proba = float(pipeline.predict_proba(x_row)[:, 1][0])
        decision = "DECLINE" if proba >= threshold else "APPROVE"

        c1, c2 = st.columns(2)
        c1.metric("Probability of default", f"{proba:.1%}")
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
                f"chance of default (decision cutoff {threshold:.0%})."
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
                "Bars to the right push the applicant toward higher default "
                "risk; bars to the left toward lower risk. "
                + ("(Approximate: global feature importances - install `shap` "
                   "for exact per-applicant values on this model.)"
                   if approximate else
                   "(Exact contributions to the model's log-odds.)")
            )
        except Exception as e:
            st.info(f"Explanation unavailable: {e}")


if __name__ == "__main__":
    main()
