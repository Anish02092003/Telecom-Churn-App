# 📡 Telecom Customer Churn Prediction (End-to-End ML Project)

## 📌 Overview
Customer churn is a major challenge in the telecom industry, directly impacting revenue and customer lifetime value.  
This project builds an **end-to-end Machine Learning system** to predict whether a telecom customer is likely to churn based on demographic details, service usage, and billing information.

The solution covers the **complete ML lifecycle** — from data preprocessing and model training to deployment using Flask and cloud hosting.

---

## 🎯 Problem Statement
Telecom companies lose significant revenue when customers discontinue their services.  
The objective of this project is to **predict customer churn in advance**, enabling businesses to take proactive retention measures such as personalized offers or service improvements.

---

## 🧠 Solution Approach
- Treated churn prediction as a **binary classification problem**
- Performed **data cleaning, feature encoding, and scaling**
- Trained multiple ML models and selected the best-performing one
- Built a **Flask web application** for real-time predictions
- Deployed the application to the cloud

---

## 📂 Project Structure
telecom-churn/
│── app.py
│── requirements.txt
│── models/
│ ├── churn_model.pkl
│ ├── scaler.pkl
│ ├── feature_names.pkl
│── src/
│ ├── churn_pipeline.py
│ ├── test_model.py
│── templates/
│ └── index.html
│── data/


---

## 📊 Dataset
- **Dataset:** Telco Customer Churn
- **Features include:**
  - Customer tenure
  - Monthly & total charges
  - Contract type
  - Internet service type
  - Billing preferences
  - Demographic attributes
- **Target variable:** `Churn` (Yes / No)

---

## ⚙️ Machine Learning Pipeline
1. Data Cleaning & Preprocessing  
2. One-Hot Encoding of Categorical Features  
3. Feature Scaling using `StandardScaler`  
4. Model Training:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
5. Model Evaluation using:
   - Precision
   - Recall
   - ROC-AUC Score
6. Best model selection and persistence

---

## 🚀 Deployment
- Backend: **Flask**
- Model serving using saved artifacts (`.pkl`)
- Cloud hosting using **Render**
- Ensured feature consistency during inference using saved feature mappings
- ## Deployed Here --
-  https://telecom-churn-app-go4q.onrender.com

---

## 🖥️ Web Application
The deployed web app allows users to:
- Enter customer details via a form
- Get real-time churn predictions
- Understand whether a customer is likely to churn or not

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn
- **Web Framework:** Flask
- **Deployment:** Render
- **Version Control:** Git & GitHub

---

## 📈 Key Learnings
- Handling categorical feature consistency between training and inference
- Building production-ready ML pipelines
- Deploying ML models as web applications
- Debugging real-world ML deployment issues

---

## 📌 Future Improvements
- Add feature importance visualization
- Improve UI/UX using Bootstrap
- Add authentication and logging
- Integrate database for prediction history

---

## 👨‍💻 Author
**Pritish Kumar Lenka**  
Electronics & Communication Engineering  
Machine Learning | Data Science | Applied AI
