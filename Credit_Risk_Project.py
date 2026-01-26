import mysql.connector
import pandas as pd

# 1. Setup the connection details
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="****",
    database="Credit_Risk_Project"
)

# 2. Check if it worked
if mydb.is_connected():
    print("Successfully connected to the database!")
else:
    print("Connection failed.")











import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
db_config = {
    'user': 'root',
    'password': '****',
    'host': 'localhost',
    'database': 'Credit_Risk_Project'
}

# ==========================================
# STEP 1: ETL (UPLOAD CSV TO MYSQL)
# ==========================================
print("--- STEP 1: UPLOADING DATA TO MYSQL ---")

# 1. Read the CSV
# Make sure the filename matches exactly what you downloaded
try:
    df = pd.read_csv('german_credit_data.csv')
    print("CSV loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Could not find 'german_credit_data.csv'.")
    print("Please move the downloaded file to this folder and try again.")
    exit()

# 2. Clean Column Names (Remove spaces/slashes for SQL compatibility)
df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')
# Expected columns: Age, Sex, Job, Housing, Saving_accounts, Checking_account, Credit_amount, Duration, Purpose, Risk

# 3. Handle Missing Values 
# We fill missing "Saving_accounts" with "unknown"
df = df.fillna('unknown')

# 4. Connect to MySQL using SQLAlchemy (Best for uploading large data)
# Format: mysql+mysqlconnector://user:password@host/database
connection_string = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
engine = create_engine(connection_string)

# 5. Push to SQL
df.to_sql('loans_raw', con=engine, if_exists='replace', index=False)
print("✅ Data successfully uploaded to MySQL table 'loans_raw'.")


# ==========================================
# STEP 2: SQL QUERY 
# ==========================================
print("\n--- STEP 2: FETCHING ANALYTICS FROM SQL ---")

# Simulate a Business Requirement:
# "The risk team only wants to model loans for Cars and Radios/TV, as these are our focus areas."
query = """
SELECT Age, Sex, Credit_amount, Duration, Purpose, Risk, Housing, Saving_accounts
FROM loans_raw
WHERE Purpose IN ('car', 'radio/TV')
"""

# Read filtered data back into Python
df_model = pd.read_sql(query, engine)
print(f"Fetched {len(df_model)} focused records from Database.")


# ==========================================
# STEP 3: MODELING 
# ==========================================
print("\n--- STEP 3: BUILDING LOGISTIC REGRESSION MODEL ---")

# 1. Preprocessing
# Machines can't read text like "male" or "bad". We must convert them to numbers.
le = LabelEncoder()
df_model['Sex_Code'] = le.fit_transform(df_model['Sex'])       # male/female -> 0/1
df_model['Risk_Code'] = le.fit_transform(df_model['Risk'])     # good/bad -> 0/1
df_model['Housing_Code'] = le.fit_transform(df_model['Housing'])    # own/free/rent -> 0/1/2
df_model['Saving_accounts_Code'] = le.fit_transform(df_model['Saving_accounts'])   # unknown/little/moderate/quite rich/rich -> 0/1/2/3/4

# 2. Define Features (X) and Target (y)
X = df_model[['Age', 'Credit_amount', 'Duration', 'Sex_Code', 'Housing_Code', 'Saving_accounts_Code']]
y = df_model['Risk_Code']

# 3. Split Data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n", classification_report(y_test, predictions))

print("\n✅ PROJECT COMPLETE!")