# =========================================================
# TELECOM CUSTOMER CHURN - END TO END PIPELINE
# =========================================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print("Dataset loaded successfully\n")
    return df


# =========================================================
# 2. CLEAN DATA
# =========================================================
def clean_data(df):
    print("Cleaning data...")

    df = df.copy()

    # Drop customerID (no predictive value)
    df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges column
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Binary columns
    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Internet related columns
    internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    for col in internet_cols:
        df[col] = df[col].replace({
            "Yes": 1,
            "No": 0,
            "No internet service": 0
        })

    print("Data cleaning completed\n")
    return df


# =========================================================
# 3. FEATURE ENCODING
# =========================================================
def encode_features(df):
    print("Encoding categorical features...")

    df_encoded = pd.get_dummies(df, drop_first=True)

    print("Encoding completed\n")
    return df_encoded


# =========================================================
# 4. SPLIT & SCALE DATA
# =========================================================
def split_and_scale(df):
    print("Splitting and scaling data...")


    X = df.drop("Churn", axis=1)
    feature_names = X.columns
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data split and scaling done\n")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names



# =========================================================
# 5. TRAIN & EVALUATE MODELS
# =========================================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("Training models...\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_auc = 0

    for name, model in models.items():
        print(f"--- {name} ---")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {auc:.4f}\n")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    print(f"Best model selected with ROC-AUC = {best_auc:.4f}\n")
    return best_model


# =========================================================
# 6. SAVE MODEL & SCALER
# =========================================================
def save_artifacts(model, scaler):
    print("Saving model and scaler...")

    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Model and scaler saved successfully\n")


# =========================================================
# 7. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    df = load_data("data/telecom_churn.csv")
    df = clean_data(df)
    df = encode_features(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    save_artifacts(best_model, scaler)
    with open("models/feature_names.pkl", "wb") as f:
     pickle.dump(feature_names, f)


    print("PIPELINE EXECUTED SUCCESSFULLY 🚀")
