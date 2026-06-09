USE Credit_Risk_Project;

SHOW TABLES;

-- Inspect the table loaded from the loan-approval CSV via etl.py
SELECT * FROM loans_raw LIMIT 100;

-- Approval vs rejection counts
SELECT loan_status, COUNT(*) AS applications
FROM loans_raw
GROUP BY loan_status;

-- Average CIBIL score by outcome (rejections cluster at low scores)
SELECT loan_status, ROUND(AVG(cibil_score)) AS avg_cibil
FROM loans_raw
GROUP BY loan_status;

-- Approval rate by education level
SELECT education,
       COUNT(*) AS applications,
       SUM(loan_status = 'Approved') AS approved,
       ROUND(100 * SUM(loan_status = 'Approved') / COUNT(*), 1) AS approval_pct
FROM loans_raw
GROUP BY education;

-- Average requested loan amount and income by outcome
SELECT loan_status,
       ROUND(AVG(income_annum)) AS avg_annual_income,
       ROUND(AVG(loan_amount)) AS avg_loan_amount
FROM loans_raw
GROUP BY loan_status;

-- Self-employed split
SELECT self_employed, COUNT(*) AS applications
FROM loans_raw
GROUP BY self_employed;
