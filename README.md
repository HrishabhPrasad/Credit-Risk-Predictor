# Credit Risk Predictor

An end-to-end, explainable **loan-approval / credit-eligibility** model for the
Indian lending context. It ingests applicant records into a **MySQL** warehouse,
trains an imbalance-aware classifier with a **cost-sensitive decision
threshold**, explains every prediction (CIBIL score, income, assets, …), and
serves it through an interactive **Streamlit** app.

**Dataset:** [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
(`loan_approval_dataset.csv`) — 4,269 Indian loan applications with CIBIL score,
annual income, asset values, loan amount/term, and the bank's
`loan_status` (Approved / Rejected).

> **What the model predicts:** whether an application would be **Approved or
> Rejected** (eligibility), not real-world default. The target is the lender's
> historical decision. Frame it as an automated *loan-eligibility* screener.

---

## Methodology

- **Correct preprocessing.** Categorical fields (`education`, `self_employed`)
  are one-hot encoded; numerics are scaled. Identifier columns (`loan_id`) and
  any protected attributes (`Sex`/`Gender`) are dropped so decisions never rest
  on them.
- **Model selection.** Logistic Regression vs. Random Forest, chosen by
  stratified 5-fold cross-validated ROC-AUC.
- **Class imbalance** handled with `class_weight="balanced"` (~38% of
  applications are rejected).
- **Cost-sensitive threshold.** Wrongly approving an applicant who should be
  declined is treated as **5x** as costly as wrongly declining a sound one; the
  threshold minimises expected cost on cross-validated predictions, not the
  naive 0.50 cutoff.
- **Explainability.** A SHAP summary (global) plus per-applicant, plain-English
  factor lists in the app ("CIBIL score = 820 reduced rejection risk").

**Result:** test **ROC-AUC ≈ 0.997**.

> **Honesty note on the score.** This dataset's `loan_status` is almost entirely
> determined by `cibil_score` (applications below ~550 are nearly always
> rejected), so it is close to linearly separable and any reasonable model
> scores very high. Treat the near-perfect accuracy as a property of the data,
> not evidence of a hard problem solved. The value of the project is the
> end-to-end, explainable, production-style pipeline — not the headline number.

---

## Project layout

```
etl.py            CSV -> MySQL -> DataFrame (CSV fallback for no-DB runs)
train_model.py    Preprocessing, model selection, threshold tuning, SHAP
app.py            Streamlit app: live prediction + plain-English explanation
analysis_queries.sql  Exploratory SQL against the warehouse table
loan_approval_dataset.csv  The dataset
models/           Saved pipeline + metadata + SHAP background (generated)
reports/          Confusion matrix + SHAP summary plots (generated)
```

---

## Quick start

```bash
pip install -r requirements.txt
```

### Option A — MySQL (primary pipeline)

```bash
export CREDIT_DB_USER=root
export CREDIT_DB_PASSWORD=yourpassword
export CREDIT_DB_HOST=localhost
export CREDIT_DB_NAME=Credit_Risk_Project

python etl.py            # load the CSV into MySQL
python train_model.py    # train (reads from MySQL)
```

### Option B — no database (CSV fallback)

```bash
python train_model.py --source csv
```

### Run the app

```bash
streamlit run app.py
```

The app loads the committed model, so it needs no database. Enter an
applicant's details and it returns the rejection probability, an
APPROVE / DECLINE decision at the cost-tuned threshold, and the factors that
drove it.

---

## Using a different dataset

The pipeline auto-detects numeric vs. categorical columns, so any tabular
credit dataset works:

```bash
python train_model.py --source csv --csv YOURFILE.csv \
    --target TARGET_COLUMN --positive-label RISKY_VALUE
```

---

## Possible next steps

- Calibrated probabilities (`CalibratedClassifierCV`) for reliable scoring.
- Validate on a dataset with genuine default labels (not just approval
  decisions) to test real predictive power.
