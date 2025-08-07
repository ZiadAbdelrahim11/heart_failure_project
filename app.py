import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define prediction function
def predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time):
    # Create DataFrame from inputs
    input_data = pd.DataFrame([[
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
        sex, smoking, time
    ]], columns=[
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
        'sex', 'smoking', 'time'
    ])

    # Scale required columns
    cols_to_scale = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction',
        'platelets', 'serum_creatinine', 'serum_sodium', 'time'
    ]
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Make prediction
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    return f"Prediction: {'⚠️ At Risk' if pred == 1 else '✅ Not At Risk'} (Probability of Death: {proba:.2f})"

# Define Gradio interface
interface = gr.Interface(
    fn=predict_heart_failure,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Anaemia (0 = No, 1 = Yes)"),
        gr.Number(label="Creatinine Phosphokinase"),
        gr.Radio([0, 1], label="Diabetes (0 = No, 1 = Yes)"),
        gr.Number(label="Ejection Fraction"),
        gr.Radio([0, 1], label="High Blood Pressure (0 = No, 1 = Yes)"),
        gr.Number(label="Platelets"),
        gr.Number(label="Serum Creatinine"),
        gr.Number(label="Serum Sodium"),
        gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
        gr.Radio([0, 1], label="Smoking (0 = No, 1 = Yes)"),
        gr.Number(label="Follow-up Time (days)")
    ],
    outputs="text",
    title="Heart Failure Risk Prediction",
    description="Enter patient data to predict the risk of death from heart failure."
)

if __name__ == "__main__":
    interface.launch()
