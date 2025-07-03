import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("smart_health_rf_model.pkl")
scaler = joblib.load("smart_health_scaler.pkl")
label_encoder = joblib.load("symptoms_label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")  # Ensures correct column order

st.set_page_config(page_title="Smart Health Prediction", layout="centered")
st.title("Smart Health Prediction System")

with st.form("health_form"):
    st.subheader("Enter Health Details")
    heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 75)
    spo2 = st.slider("SpO2 (%)", 80, 100, 95)
    temperature = st.slider("Temperature (°C)", 35.0, 42.0, 37.0)
    glucose = st.slider("Glucose Level (mg/dL)", 70, 300, 100)
    steps = st.number_input("Steps Count", min_value=0, value=5000)

    systolic = st.slider("Systolic BP", 80, 200, 120)
    diastolic = st.slider("Diastolic BP", 50, 130, 80)

    device_location = st.selectbox("Device Location", ["Clinic", "Hospital", "Home"])
    wearable = st.radio("Using Wearable?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # One-hot encode location manually
    location_clinic = 1 if device_location == "Clinic" else 0
    location_hospital = 1 if device_location == "Hospital" else 0
    location_home = 1 if device_location == "Home" else 0

    wearable_encoded = 1 if wearable == "Yes" else 0

    # Prepare input
    input_data = pd.DataFrame([[
        heart_rate, spo2, temperature, glucose, steps,
        wearable_encoded, systolic, diastolic,
        location_clinic, location_hospital, location_home
    ]], columns=[
        "HeartRate (bpm)", "SpO2 (%)", "Temperature (°C)", "GlucoseLevel (mg/dL)", "Steps",
        "UsingWearable", "SystolicBP", "DiastolicBP",
        "DeviceLocation_Clinic", "DeviceLocation_Hospital", "DeviceLocation_Home"
    ])

    # Reorder to match training-time feature order
    input_data = input_data[feature_names]

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Symptom: **{predicted_label}**")
