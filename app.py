"""
app.py
======
Streamlit front-end for the Credit Risk Predictor.

A loan officer enters an applicant's details, and the app returns:
  * the model's estimated probability of default,
  * an APPROVE / DECLINE decision using the cost-optimised threshold, and
  * a SHAP explanation showing which factors pushed the decision each way.

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
    pipeline = joblib.load(MODELS_DIR / "credit_risk_model.joblib")
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


def explain(pipeline, background: pd.DataFrame, x_row: pd.DataFrame, model_name: str):
    """Return (feature_names, shap_values) for the single applicant."""
    import shap

    pre = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    feat_names = pre.get_feature_names_out()

    X_bg = pre.transform(background)
    X_bg = X_bg.toarray() if hasattr(X_bg, "toarray") else X_bg
    X_row = pre.transform(x_row)
    X_row = X_row.toarray() if hasattr(X_row, "toarray") else X_row

    if model_name == "RandomForest":
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
    else:
        explainer = shap.LinearExplainer(clf, X_bg)
        sv = explainer.shap_values(X_row)
    return feat_names, np.asarray(sv)[0]


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
        st.metric("Algorithm", meta["model"])
        st.metric("Test ROC-AUC", f"{meta['test_roc_auc']:.3f}")
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

        st.subheader("Why? (SHAP contributions)")
        try:
            names, sv = explain(pipeline, background, x_row, meta["model"])
            contrib = (
                pd.DataFrame({"feature": names, "shap_value": sv})
                .assign(abs_val=lambda d: d["shap_value"].abs())
                .sort_values("abs_val", ascending=False)
                .head(10)
                .set_index("feature")["shap_value"]
            )
            st.bar_chart(contrib)
            st.caption(
                "Positive values push the applicant toward higher default risk; "
                "negative values toward lower risk."
            )
        except Exception as e:
            st.info(f"Explanation unavailable: {e}")


if __name__ == "__main__":
    main()
