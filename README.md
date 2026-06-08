# Credit Risk Predictor

An end-to-end, explainable credit-default risk pipeline. It ingests raw loan
records into a **MySQL** warehouse, trains an imbalance-aware classifier with a
**cost-sensitive decision threshold**, explains every prediction with **SHAP**,
and serves it through an interactive **Streamlit** app.

The pipeline is dataset-agnostic and ships with two datasets:

| Dataset | File | Source | Use |
|---|---|---|---|
| German Credit (benchmark) | `german_credit_data.csv` | Public UCI benchmark | Primary / default |
| Indian lending context | `indian_credit_data.csv` | **Synthetic** (generated) | Domain-relevance demo |

---

## Why this is more than a logistic regression

The original version trained logistic regression on 6 features of a single loan
purpose and reported 68.55% accuracy - a number dominated by class imbalance. It
collapsed into a "paranoid banker": 95% recall on bad loans, 12% on good ones.
This version fixes the methodology:

- **Correct preprocessing.** Nominal categories (Housing, Purpose, ...) are
  one-hot encoded, not label-encoded - removing the false ordinality
  (`own=0 < rent=2`) the original code introduced. Numerics are scaled.
- **Full data.** Uses every loan and the full feature set (the original silently
  dropped all purposes except `car` / `radio/TV`).
- **Imbalance handling.** `class_weight="balanced"` instead of letting the
  majority class dominate.
- **Cost-sensitive threshold.** Approving a defaulter (false negative) is treated
  as **5x** as costly as declining a good applicant (false positive) - the
  documented German Credit cost convention. The decision threshold is chosen to
  minimise expected cost on cross-validated predictions, not left at a naive 0.50.
- **Honest evaluation.** Stratified 5-fold CV ROC-AUC for model selection, plus
  test-set ROC-AUC, PR-AUC, confusion matrix and a full classification report.
- **Explainability.** SHAP summary (global) and per-applicant contributions (in
  the app) - the part interviewers actually probe.

On the German benchmark this yields **ROC-AUC ≈ 0.76** and, at the cost-tuned
threshold, **~0.90 recall on defaulters** - a deliberate, defensible tradeoff
rather than an accidental one.

---

## Project layout

```
etl.py                     CSV -> MySQL -> DataFrame (CSV fallback for no-DB runs)
train_model.py             Preprocessing, model selection, threshold tuning, SHAP
app.py                     Streamlit app: live prediction + SHAP explanation
generate_indian_dataset.py Synthetic Indian-context dataset generator
analysis_queries.sql       Exploratory SQL against the warehouse table
models/                    Saved pipeline + metadata + SHAP background (generated)
reports/                   Confusion matrix + SHAP summary plots (generated)
```

---

## Quick start

```bash
pip install -r requirements.txt
```

### Option A - MySQL (primary pipeline)

```bash
# 1. Configure credentials (no secrets in source)
export CREDIT_DB_USER=root
export CREDIT_DB_PASSWORD=yourpassword
export CREDIT_DB_HOST=localhost
export CREDIT_DB_NAME=Credit_Risk_Project

# 2. Load the CSV into MySQL
python etl.py

# 3. Train (reads from MySQL)
python train_model.py
```

### Option B - no database (CSV fallback)

```bash
python train_model.py --source csv
```

### Run the app

```bash
streamlit run app.py
```

---

## Indian lending context

`generate_indian_dataset.py` produces a dataset framed around Indian lending -
CIBIL score, monthly income (INR), EMIs, employment type, city tier - to
demonstrate domain relevance to Indian BFSI recruiters:

```bash
python generate_indian_dataset.py
python train_model.py --source csv --csv indian_credit_data.csv
```

> **Honesty note.** `indian_credit_data.csv` is **synthetic** - generated from
> hand-coded relationships, not real borrowers. Use it to demonstrate
> *methodology*, not as evidence of real-world predictive power. For a genuine
> differentiator, drop in real data (CMIE Prowess, RBI/CIBIL public aggregates,
> or a Kaggle Indian lending dataset) using the same column schema and the same
> commands. Never present synthetic results as real findings.

---

## Possible next steps

- Calibrated probabilities (`CalibratedClassifierCV`) for reliable risk scoring.
- Reject-inference / survival framing for through-the-door applicants.
- Swap in real Indian data and re-benchmark.
