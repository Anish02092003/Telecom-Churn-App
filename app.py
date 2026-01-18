from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ===============================
# LOAD MODEL ARTIFACTS
# ===============================
model = pickle.load(open("models/churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
feature_names = pickle.load(open("models/feature_names.pkl", "rb"))

# ===============================
# HOME ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        # ---------------------------
        # RAW USER INPUTS
        # ---------------------------
        tenure = int(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        total_charges = float(request.form["total_charges"])

        contract = request.form["contract"]
        internet = request.form["internet"]
        paperless = int(request.form["paperless"])
        partner = int(request.form["partner"])
        dependents = int(request.form["dependents"])
        senior = int(request.form["senior"])

        # ---------------------------
        # CREATE INPUT DATAFRAME
        # ---------------------------
        input_df = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)

        # Numerical features
        input_df.loc[0, "tenure"] = tenure
        input_df.loc[0, "MonthlyCharges"] = monthly_charges
        input_df.loc[0, "TotalCharges"] = total_charges
        input_df.loc[0, "PaperlessBilling"] = paperless
        input_df.loc[0, "Partner"] = partner
        input_df.loc[0, "Dependents"] = dependents
        input_df.loc[0, "SeniorCitizen"] = senior

        # One-hot encoded categorical features
        contract_col = f"Contract_{contract}"
        internet_col = f"InternetService_{internet}"

        if contract_col in input_df.columns:
            input_df.loc[0, contract_col] = 1

        if internet_col in input_df.columns:
            input_df.loc[0, internet_col] = 1

        # ---------------------------
        # SCALE + PREDICT
        # ---------------------------
        scaled_input = scaler.transform(input_df)
        result = model.predict(scaled_input)[0]

        prediction = "Customer will CHURN ❌" if result == 1 else "Customer will NOT churn ✅"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

