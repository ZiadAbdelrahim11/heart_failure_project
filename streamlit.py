# src/app.py

import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="🩺",
    layout="centered"
)

# Feature names in the correct order
feature_names = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# Columns to scale (must match scaler training)
cols_to_scale = [
    'age',
    'creatinine_phosphokinase',
    'ejection_fraction',
    'platelets',
    'serum_creatinine',
    'serum_sodium',
    'time'
]

# UI Title and description
st.title("🩺 Heart Failure Prediction App")
st.markdown("This app predicts the likelihood of a mortality event based on patient clinical records. Please enter the patient's data in the sidebar.")

# Sidebar inputs
st.sidebar.header("Patient Data Input")

age = st.sidebar.slider("Age", 40, 95, 60)
sex = st.sidebar.radio("Sex", ("Male", "Female"))
anaemia = st.sidebar.radio("Anaemia", ("No", "Yes"))
diabetes = st.sidebar.radio("Diabetes", ("No", "Yes"))
high_blood_pressure = st.sidebar.radio("High Blood Pressure", ("No", "Yes"))
smoking = st.sidebar.radio("Smoking", ("No", "Yes"))

st.sidebar.divider()

cpk = st.sidebar.number_input("Creatine Phosphokinase (CPK) [mcg/L]", value=582, step=1)
ejection_fraction = st.sidebar.slider("Ejection Fraction [%]", 14, 80, 38)
platelets = st.sidebar.number_input("Platelets [kiloplatelets/mL]", value=263358.0)
serum_creatinine = st.sidebar.slider("Serum Creatinine [mg/dL]", 0.5, 9.4, 1.1, 0.1)
serum_sodium = st.sidebar.slider("Serum Sodium [mEq/L]", 113, 148, 136)
time = st.sidebar.number_input("Follow-up Period [days]", value=130, step=1)

predict_btn = st.sidebar.button("Predict Risk", type="primary")

if model and scaler and predict_btn:
    # Map categorical to numeric
    sex_val = 1 if sex == "Male" else 0
    anaemia_val = 1 if anaemia == "Yes" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    hbp_val = 1 if high_blood_pressure == "Yes" else 0
    smoking_val = 1 if smoking == "Yes" else 0

    # Build input dataframe
    input_data = pd.DataFrame([[
        age, anaemia_val, cpk, diabetes_val, ejection_fraction, hbp_val,
        platelets, serum_creatinine, serum_sodium, sex_val, smoking_val, time
    ]], columns=feature_names)

    # Copy and scale only relevant columns
    input_scaled = input_data.copy()
    input_scaled[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.divider()
    st.header("Prediction Result")

    if prediction[0] == 1:
        st.warning("Prediction: At-Risk (High probability of a mortality event)")
    else:
        st.success("Prediction: Not At-Risk (Low probability of a mortality event)")

    prob_at_risk = prediction_proba[0][1]
    prob_not_at_risk = prediction_proba[0][0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of Being 'At-Risk'", f"{prob_at_risk:.2%}")
    with col2:
        st.metric("Probability of Being 'Not At-Risk'", f"{prob_not_at_risk:.2%}")

    st.progress(float(prob_at_risk))

elif not model or not scaler:
    st.warning("Cannot make a prediction because the model artifacts are not loaded.")
