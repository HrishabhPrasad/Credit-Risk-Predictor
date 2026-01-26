USE Credit_Risk_Project;


SHOW TABLES;


-- Check the table you just created from Python
SELECT * FROM loans_raw LIMIT 1500;

-- Calculate average credit amount by Risk type
SELECT Risk, AVG(Credit_amount) as Avg_Loan_Size
FROM loans_raw
GROUP BY Risk;


SELECT Sex, COUNT(*)
FROM loans_raw
GROUP BY Sex; 


SELECT DISTINCT Housing FROM loans_raw;
SELECT DISTINCT Saving_accounts FROM loans_raw;