"""
train_model.py
==============
End-to-end training pipeline for the Credit Risk Predictor.

What this does that the original script did not
-----------------------------------------------
1.  Correct preprocessing. Nominal categories (Housing, Purpose, ...) are
    one-hot encoded instead of label-encoded, removing the false ordinality
    the original code introduced (e.g. own=0 < rent=2). Numerics are scaled.
2.  Uses the FULL feature set and the FULL dataset (the original silently
    dropped every loan that was not 'car' or 'radio/TV').
3.  Handles class imbalance with `class_weight="balanced"` instead of letting
    the model collapse into the "paranoid banker" that rejects everyone.
4.  Cost-sensitive decision threshold. In the German Credit cost convention,
    approving a bad borrower (false negative) is 5x as expensive as rejecting
    a good one (false positive). We pick the threshold that minimises expected
    cost on cross-validated predictions, not the naive 0.5 cutoff.
5.  Honest evaluation: stratified K-fold ROC-AUC, plus test-set ROC-AUC,
    PR-AUC, confusion matrix and a classification report at the tuned cutoff.
6.  SHAP explainability so every prediction can be justified - the thing
    interviewers actually probe.

The script is dataset-agnostic: point it at the German benchmark or at the
synthetic Indian dataset (see generate_indian_dataset.py) and it adapts the
preprocessing to whatever numeric/categorical columns are present.

Usage
-----
    python train_model.py                       # MySQL source, German data
    python train_model.py --source csv          # CSV fallback, no DB needed
    python train_model.py --source csv --csv indian_credit_data.csv \
        --positive-label bad
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: render plots to file, never to a screen
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Cost convention: a false negative (approving a borrower who defaults) is far
# more expensive than a false positive (declining a good borrower).
COST_FN = 5.0
COST_FP = 1.0


# --------------------------------------------------------------------------- #
# Data + preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """Auto-detect numeric vs categorical columns and build a ColumnTransformer."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return pre, numeric, categorical


def make_models() -> dict[str, Pipeline]:
    """Two candidate classifiers, both imbalance-aware, wrapped with the
    preprocessor. (Preprocessor is attached per-fit in `train`.)"""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


# --------------------------------------------------------------------------- #
# Threshold tuning
# --------------------------------------------------------------------------- #
def optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Return the probability threshold minimising expected misclassification
    cost, and that minimum cost."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = COST_FN * fn + COST_FP * fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t), float(best_cost)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def shap_summary(pipeline: Pipeline, X_train: pd.DataFrame, model_name: str) -> None:
    """Save a SHAP summary (beeswarm) plot for the fitted pipeline."""
    import shap

    pre = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    feat_names = pre.get_feature_names_out()
    # Use a modest background sample for speed.
    bg = X_train.sample(min(200, len(X_train)), random_state=42)
    X_bg = pre.transform(bg)
    X_bg = X_bg.toarray() if hasattr(X_bg, "toarray") else X_bg

    if model_name == "RandomForest":
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_bg)
        # Newer SHAP returns (n, features, classes); take the positive class.
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
    else:
        explainer = shap.LinearExplainer(clf, X_bg)
        sv = explainer.shap_values(X_bg)

    plt.figure()
    shap.summary_plot(sv, X_bg, feature_names=feat_names, show=False)
    plt.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_confusion(y_true, y_pred, path: Path) -> None:
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["good (0)", "bad (1)"],
        yticklabels=["good (0)", "bad (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (tuned threshold)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def train(source: str, csv_path: str, target: str, positive_label: str) -> None:
    from etl import fetch_data

    df = fetch_data(source=source, csv_path=csv_path)
    if target not in df.columns:
        raise SystemExit(
            f"Target column '{target}' not found. Available: {list(df.columns)}"
        )

    # Encode the target so the positive class (the risky event) == 1.
    y = (df[target].astype(str).str.lower() == positive_label.lower()).astype(int)
    X = df.drop(columns=[target])
    print(f"Loaded {len(df)} rows | positive (bad) rate = {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Model selection by cross-validated ROC-AUC ---------------------- #
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, clf in make_models().items():
        pre, _, _ = build_preprocessor(X_train)
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        proba_cv = cross_val_predict(
            pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        auc = roc_auc_score(y_train, proba_cv)
        results[name] = auc
        print(f"  {name:<20} CV ROC-AUC = {auc:.3f}")

    best_name = max(results, key=results.get)
    print(f"Selected model: {best_name}")

    # --- Fit best model on full training set ----------------------------- #
    pre, numeric, categorical = build_preprocessor(X_train)
    best_clf = make_models()[best_name]
    pipeline = Pipeline([("prep", pre), ("clf", best_clf)])
    pipeline.fit(X_train, y_train)

    # --- Tune the decision threshold on cross-validated train probs ------ #
    pre_t, _, _ = build_preprocessor(X_train)
    proba_train_cv = cross_val_predict(
        Pipeline([("prep", pre_t), ("clf", make_models()[best_name])]),
        X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    threshold, _ = optimal_threshold(y_train.to_numpy(), proba_train_cv)
    print(f"Cost-optimal threshold (FN cost {COST_FN}x FP): {threshold:.3f}")

    # --- Evaluate on the held-out test set ------------------------------- #
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    pred_default = (proba_test >= 0.5).astype(int)
    pred_tuned = (proba_test >= threshold).astype(int)

    test_auc = roc_auc_score(y_test, proba_test)
    test_ap = average_precision_score(y_test, proba_test)
    print(f"\nTest ROC-AUC = {test_auc:.3f} | PR-AUC = {test_ap:.3f}")
    print("\n--- Default threshold (0.50) ---")
    print(classification_report(y_test, pred_default,
                                target_names=["good", "bad"]))
    print(f"--- Tuned threshold ({threshold:.2f}) ---")
    print(classification_report(y_test, pred_tuned,
                                target_names=["good", "bad"]))

    # --- Artifacts ------------------------------------------------------- #
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "credit_risk_model.joblib")

    # Feature metadata drives the Streamlit input form.
    feature_meta = {"numeric": {}, "categorical": {}, "order": list(X.columns)}
    for c in numeric:
        feature_meta["numeric"][c] = {
            "min": float(X[c].min()),
            "max": float(X[c].max()),
            "median": float(X[c].median()),
        }
    for c in categorical:
        feature_meta["categorical"][c] = sorted(X[c].astype(str).unique().tolist())

    metadata = {
        "model": best_name,
        "target": target,
        "positive_label": positive_label,
        "threshold": threshold,
        "cv_roc_auc": results,
        "test_roc_auc": test_auc,
        "test_pr_auc": test_ap,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "features": feature_meta,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Background sample for the app's SHAP explainer.
    X_train.sample(min(200, len(X_train)), random_state=42).to_csv(
        MODELS_DIR / "background.csv", index=False
    )

    save_confusion(y_test, pred_tuned, REPORTS_DIR / "confusion_matrix.png")
    try:
        shap_summary(pipeline, X_train, best_name)
        print("Saved SHAP summary -> reports/shap_summary.png")
    except Exception as e:  # SHAP is best-effort; never block training on it
        print(f"[warn] SHAP summary skipped: {e}")

    print("\nArtifacts written to models/ and reports/. Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the credit risk model.")
    p.add_argument("--source", choices=["mysql", "csv"], default="mysql",
                   help="Data source (default: mysql, the primary pipeline).")
    p.add_argument("--csv", default="german_credit_data.csv",
                   help="CSV path used by the csv source or for MySQL loading.")
    p.add_argument("--target", default="Risk", help="Target column name.")
    p.add_argument("--positive-label", default="bad",
                   help="Value of the target that denotes the risky/default class.")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train(a.source, a.csv, a.target, a.positive_label)
