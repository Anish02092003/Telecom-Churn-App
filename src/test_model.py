import pickle
import pandas as pd
import numpy as np

# Load artifacts
model = pickle.load(open("models/churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
feature_names = pickle.load(open("models/feature_names.pkl", "rb"))

# Create empty dataframe with correct columns
input_df = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)

# Example customer (RAW values)
customer = {
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "MonthlyCharges": 75.5,
    "TotalCharges": 900,
    "PaperlessBilling": 1,
    "InternetService_Fiber optic": 1,
    "Contract_Month-to-month": 1
}

# Fill defaults with 0
input_df.loc[0] = 0

# Fill actual values
for key, value in customer.items():
    if key in input_df.columns:
        input_df.loc[0, key] = float(value)

# Scale & predict
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

print("Prediction:", "CHURN ❌" if prediction == 1 else "NO CHURN ✅")
