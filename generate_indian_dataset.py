"""
generate_indian_dataset.py
==========================
Generate a SYNTHETIC, Indian-context credit dataset for the predictor.

WHY THIS EXISTS
---------------
The German Credit benchmark is used by thousands of student projects, so it is
not a differentiator. This script produces a dataset framed around the Indian
lending context - CIBIL scores, monthly income in INR, EMIs, employment type,
city tier - so the same modelling pipeline can demonstrate domain relevance to
Indian BFSI recruiters (HDFC, Fullerton, Scienaptic, CAMS, etc.).

HONESTY NOTE (read before putting this on a resume)
---------------------------------------------------
This data is SYNTHETIC. It is generated from hand-coded relationships, not real
borrowers, so the model's accuracy on it reflects those rules, not real-world
predictive power. Use it to demonstrate METHODOLOGY (pipeline, imbalance
handling, threshold tuning, explainability). For a genuine differentiator,
replace it with real data - e.g. CMIE Prowess, RBI/CIBIL public aggregates, or
a Kaggle Indian lending dataset - keeping the same column schema. Never claim
synthetic results as real findings in an interview.

The output schema matches what train_model.py expects: a `Risk` target with
values 'good'/'bad'.

    python generate_indian_dataset.py            # -> indian_credit_data.csv
    python train_model.py --source csv --csv indian_credit_data.csv
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate(n: int = 5000) -> pd.DataFrame:
    age = RNG.integers(21, 65, n)
    gender = RNG.choice(["male", "female"], n, p=[0.62, 0.38])
    employment = RNG.choice(
        ["salaried", "self_employed", "business", "unemployed"],
        n, p=[0.55, 0.25, 0.15, 0.05],
    )
    city_tier = RNG.choice(["tier_1", "tier_2", "tier_3"], n, p=[0.4, 0.35, 0.25])
    purpose = RNG.choice(
        ["home", "car", "education", "personal", "business", "consumer_durable"],
        n, p=[0.20, 0.18, 0.12, 0.30, 0.12, 0.08],
    )

    # Monthly income (INR) varies by employment and city tier.
    base_income = {"salaried": 55000, "self_employed": 48000,
                   "business": 70000, "unemployed": 12000}
    tier_mult = {"tier_1": 1.35, "tier_2": 1.0, "tier_3": 0.75}
    income = np.array([
        base_income[e] * tier_mult[t] * RNG.lognormal(0, 0.35)
        for e, t in zip(employment, city_tier)
    ]).round(-2).astype(int)

    existing_loans = RNG.integers(0, 5, n)
    existing_emi = (income * RNG.uniform(0, 0.35, n) * (existing_loans > 0)).round(-2).astype(int)

    loan_amount = (income * RNG.uniform(3, 24, n)).round(-3).astype(int)
    tenure = RNG.choice([12, 24, 36, 48, 60, 84, 120], n)

    # CIBIL: higher for older, salaried, fewer existing loans (with noise).
    cibil = (
        650
        + (age - 21) * 1.2
        + np.where(employment == "salaried", 40, 0)
        + np.where(employment == "unemployed", -120, 0)
        - existing_loans * 18
        + RNG.normal(0, 45, n)
    ).clip(300, 900).round().astype(int)

    # Latent default risk: high DTI, low CIBIL, long tenure, unemployment.
    new_emi = loan_amount / tenure
    dti = (existing_emi + new_emi) / np.maximum(income, 1)
    logit = (
        -1.2
        + 3.5 * (dti - 0.4)
        + 0.010 * (700 - cibil)
        + 0.004 * (tenure - 36)
        + np.where(employment == "unemployed", 1.5, 0)
        + np.where(city_tier == "tier_3", 0.3, 0)
        + RNG.normal(0, 0.5, n)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    risk = np.where(RNG.uniform(0, 1, n) < prob_default, "bad", "good")

    return pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Employment_Type": employment,
        "City_Tier": city_tier,
        "Monthly_Income_INR": income,
        "Existing_Loans": existing_loans,
        "Existing_EMI_INR": existing_emi,
        "Loan_Amount_INR": loan_amount,
        "Loan_Tenure_Months": tenure,
        "CIBIL_Score": cibil,
        "Purpose": purpose,
        "Risk": risk,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=5000, help="Number of synthetic rows.")
    p.add_argument("-o", "--out", default="indian_credit_data.csv")
    args = p.parse_args()

    df = generate(args.n)
    df.to_csv(args.out, index=False)
    rate = (df["Risk"] == "bad").mean()
    print(f"Wrote {len(df)} SYNTHETIC rows to {args.out} | bad rate = {rate:.1%}")
