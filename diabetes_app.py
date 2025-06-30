import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("diabetesmodel.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Diabetes Risk Assessment", page_icon="🏥", layout="centered")
st.title("🩺 Diabetes Risk Assessment")
st.markdown("Provide the information below to assess your diabetes risk.")

# Input: Name
name = st.text_input("👤 Name", help="Please enter your name.")

# Input: Gender and dynamic Pregnancy field
gender = st.selectbox("🚻 Gender", ["Female", "Male"], help="Used to determine whether to prompt pregnancy input.")
if gender == "Female":
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, help="Enter how many times you have been pregnant.")
else:
    pregnancies = 0  # Default for males

# Input: Age
age = st.number_input("Age", min_value=1, max_value=120, step=1, help="Enter your age in years.")

# Input: Glucose
glucose = st.slider("🧪 Glucose Level (mg/dL)", min_value=0, max_value=200, value=120, help="Normal fasting glucose is around 70-100 mg/dL.")

# Input: Blood Pressure
blood_pressure = st.slider("🫀 Blood Pressure (systolic)", min_value=40, max_value=200, value=80, help="Normal blood pressure is usually around 120/80 mmHg. Enter the systolic (top) value.")

# Input: Height and Weight to compute BMI
height = st.number_input("📏 Height (in meters)", min_value=0.5, max_value=2.5, step=0.01, help="E.g., 1.75")
weight = st.number_input("⚖️ Weight (in kilograms)", min_value=10.0, max_value=300.0, step=0.1, help="E.g., 70.5")

# Backend default values for hidden features
insulin = 80  # Default insulin value
skin_thickness = 20  # Hidden from user
diabetes_pedigree = 0.5  # Hidden from user

# Input validation: Prevent empty name or zero values
if st.button("Predict Diabetes Risk"):
    if name.strip() == "":
        st.warning("Please enter your name before submitting.")
    elif height == 0:
        st.warning("Height cannot be zero.")
    else:
        # Calculate BMI
        bmi = weight / (height ** 2)

        # Combine all features into the correct order expected by the model
        input_data = np.array([
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, diabetes_pedigree, age
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Show result
        st.subheader(f"Hi {name}, here is your result:")
        if prediction == 1:
            st.error("⚠️ Based on the input, you are likely to have diabetes.")
        else:
            st.success("✅ Based on the input, you are not likely to have diabetes.")

# Footer
st.markdown("---")
st.markdown("App created for academic purposes | Asia Pacific University")