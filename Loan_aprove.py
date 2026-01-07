import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")   # keep file in same folder

# Drop Loan_ID (not useful for prediction)
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# ---------------------------
# Handle Missing Values
# ---------------------------
# Fill categorical NaN with mode (most frequent)
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numeric NaN with median
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# ---------------------------
# Encode Categorical Features
# ---------------------------
label_encoders = {}
for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target encoding
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["Loan_Status"])   # Y/N → 1/0
X = df.drop("Loan_Status", axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("🏦 Loan Approval Prediction App")

st.markdown("Enter applicant details below to check **Loan Status**")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=2000)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
    loan_term = st.selectbox("Loan Term (in days)", [360, 180, 120, 60])
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Loan Approval"):
    input_data = pd.DataFrame({
        "Gender": [1 if gender == "Male" else 0],
        "Married": [1 if married == "Yes" else 0],
        "Dependents": [0 if dependents=="0" else 1 if dependents=="1" else 2 if dependents=="2" else 3],
        "Education": [0 if education == "Graduate" else 1],
        "Self_Employed": [1 if self_employed == "Yes" else 0],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [0 if property_area=="Urban" else 1 if property_area=="Semiurban" else 2]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.subheader("🔍 Prediction Result:")
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")


# LP001028,Male,Yes,2,Graduate,No,3073,8106,200,360,1,Urban,Y
# LP001036,Female,No,0,Graduate,No,3510,0,76,360,0,Urban,N
