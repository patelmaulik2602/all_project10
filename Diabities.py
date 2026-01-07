
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ------------------------
# Load Dataset
# ------------------------
df = pd.read_csv("diabetes.csv")

x = df.drop(columns="Outcome", axis=1)
y = df["Outcome"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# ------------------------
# Streamlit UI
# ------------------------
st.title("🩺 Diabetes Prediction App")

# Create 2 columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Patient Details")
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    Glucose = st.number_input("Glucose", min_value=50, max_value=200, value=120)
    BloodPressure = st.number_input("Blood Pressure", min_value=40, max_value=122, value=70)
    SkinThickness = st.number_input("Skin Thickness", min_value=10, max_value=100, value=20)

with col2:
    st.subheader("")

    Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    BMI = st.number_input("BMI", min_value=10, max_value=70, value=25)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    Age = st.number_input("Age", min_value=1, max_value=100, value=30)

# Predict Button
if st.button("🔍 Predict"):
    # Convert input to numpy array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    result = "✅ Non-Diabetic" if prediction[0] == 0 else "⚠️ Diabetic"

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.success(result)
