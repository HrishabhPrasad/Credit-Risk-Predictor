# 💳 Credit Risk Predictor

An **explainable loan-eligibility predictor** for the Indian lending context: enter an applicant's CIBIL score, income and assets, and get an **Approve / Decline** decision *with the reasons behind it.*

🔗 **Live demo:** https://credit-risk-predictor-dashboard.streamlit.app/  ·  **Stack:** Python · scikit-learn · SHAP · MySQL · Streamlit

---

### What it does
Trains on **4,269 real Indian loan applications** ([dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)) to predict the lender's **Approve/Reject** decision, and serves it as an interactive web app that explains every prediction.

### Highlights
- **Proper preprocessing** — one-hot encoding for categories, scaling for numerics; identifier and protected attributes (`Sex`/`Gender`) dropped for fairness.
- **Model selection** — Logistic Regression vs. Random Forest, chosen by stratified 5-fold cross-validated ROC-AUC.
- **Class imbalance** handled with balanced class weights (~38% rejections).
- **Cost-sensitive threshold** — approving a bad applicant is treated as 5× costlier than declining a good one, so the cutoff is tuned to minimise expected cost (not a naive 0.50).
- **Explainable** — per-applicant SHAP factors in plain English ("CIBIL = 820 reduced rejection risk").

**Test ROC-AUC ≈ 0.997** — note: `loan_status` is nearly determined by `cibil_score`, so the data is almost separable; the value here is the explainable, end-to-end pipeline, not the headline number.

### Run it
```bash
pip install -r requirements.txt
python train_model.py --source csv   # train (no database needed)
streamlit run app.py                 # launch the app
```
