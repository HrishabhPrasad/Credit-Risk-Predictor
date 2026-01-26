"A project to predict credit risk using Python (ETL) and MySQL"



# Credit Risk Prediction Pipeline

### Project Overview
This project develops an end-to-end data pipeline and predictive model to assess credit default risk. By simulating a real-world banking environment, the system ingests raw financial records, stores them in a relational database (MySQL), and applies a Logistic Regression model to classify loan applicants as "Good" or "Bad" credit risks.

### Business Problem
Financial institutions face significant losses when borrowers default. The goal of this project is to minimize **Type II Errors** (approving a bad borrower) by identifying high-risk applicants based on demographic and financial indicators before a loan is sanctioned.

### Methodology & Stack
* **Data Engineering (ETL):** Built an automated pipeline using **Python (Pandas)** to extract raw CSV data, clean inconsistent labeling, and load structured data into a local **MySQL** database.
* **Data Storage:** Designed a relational schema to serve as the single source of truth for loan analysis.
* **Statistical Modeling:** Implemented **Logistic Regression** (via Scikit-Learn) to predict binary outcomes (Default vs. Non-Default).

### Key Results
* **Model Accuracy:** Achieved a baseline accuracy of **68.55%**.
* **Risk Identification:** The model successfully identified key variables correlated with default risk, though further feature engineering (e.g., Debt-to-Income ratio) is recommended to improve recall.

### Next Steps
* Improve model sensitivity (Recall) to catch more potential defaulters.
* Implement Random Forest classifiers to capture non-linear relationships between age and credit amount.