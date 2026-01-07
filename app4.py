import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("diabetes.csv")  # make sure your dataset is in same folder

# Features and target
X = df.drop(columns="Outcome", axis=1)
y = df["Outcome"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardization
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model Training
ls = LogisticRegression()
ls.fit(x_train, y_train)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details to check if the person is **Diabetic** or **Non-Diabetic**.")

# Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("Predict Diabetes"):
    # Prepare input
    input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    input_data_as_numpy = np.asarray(input_data).reshape(1, -1)
    std_data = sc.transform(input_data_as_numpy)

    prediction = ls.predict(std_data)

    if prediction[0] == 0:
        st.success("✅ The person is **Non-Diabetic**")
    else:
        st.error("⚠️ The person is **Diabetic**")

# Show Model Accuracy
st.subheader("📊 Model Performance")
y_pred = ls.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy on Test Data: **{accuracy*100:.2f}%**")

