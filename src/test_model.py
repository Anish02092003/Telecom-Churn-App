import pickle
import pandas as pd
import numpy as np

model = pickle.load(open("models/churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
feature_names = pickle.load(open("models/feature_names.pkl", "rb"))

input_df = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)

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

input_df.loc[0] = 0

for key, value in customer.items():
    if key in input_df.columns:
        input_df.loc[0, key] = float(value)

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

print("Prediction:", "CHURN ❌" if prediction == 1 else "NO CHURN ✅")

