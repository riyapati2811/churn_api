import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Risk Predictor")
st.write("Enter a customer's details to estimate their likelihood of churning.")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input("Monthly Charges (₹)", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges (₹)", min_value=0.0, value=840.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Has Partner", ["No", "Yes"])
    Dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# --- Predict button ---
if st.button("Predict Churn Risk", type="primary"):
    payload = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": 1 if Partner == "Yes" else 0,
        "Dependents": 1 if Dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if PhoneService == "Yes" else 0,
        "PaperlessBilling": 1 if PaperlessBilling == "Yes" else 0,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
    }

    API_URL = "https://churn-api-enb4.onrender.com/predict"

    with st.spinner("Contacting the model... (may take a moment if the server was asleep)"):
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        prob = result["churn_probability"]
        pred = result["churn_prediction"]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ High Risk: {prob*100:.1f}% chance of churning")
        else:
            st.success(f"✅ Low Risk: {prob*100:.1f}% chance of churning")

        st.progress(prob)
    else:
        st.error(f"API error: {response.status_code} — {response.text}")