# app.py
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("🏥 Healthcare Insurance Charges Prediction App")

st.write("Fill in the patient details to predict healthcare insurance charges.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "southeast", "northwest", "southwest"])

# Encode categorical variables
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0
region_encoded = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}[region]

# Prediction
if st.button("Predict Charges"):
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Medical Charges: ${prediction[0]:,.2f}")
