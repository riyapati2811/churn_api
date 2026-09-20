from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

app = FastAPI(title="Customer Churn Prediction API")

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines: str        # "No", "Yes", "No phone service"
    InternetService: str      # "DSL", "Fiber optic", "No"
    OnlineSecurity: str       # "No", "Yes", "No internet service"
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str             # "Month-to-month", "One year", "Two year"
    PaymentMethod: str        # "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    raw = data.dict()

    # Start with the simple numeric/binary fields
    input_dict = {
        'gender': raw['gender'],
        'SeniorCitizen': raw['SeniorCitizen'],
        'Partner': raw['Partner'],
        'Dependents': raw['Dependents'],
        'tenure': raw['tenure'],
        'PhoneService': raw['PhoneService'],
        'PaperlessBilling': raw['PaperlessBilling'],
        'MonthlyCharges': raw['MonthlyCharges'],
        'TotalCharges': raw['TotalCharges'],
    }

    # Manually recreate the one-hot columns, matching training exactly
    one_hot_map = {
        'MultipleLines_No phone service': raw['MultipleLines'] == 'No phone service',
        'MultipleLines_Yes': raw['MultipleLines'] == 'Yes',
        'InternetService_Fiber optic': raw['InternetService'] == 'Fiber optic',
        'InternetService_No': raw['InternetService'] == 'No',
        'OnlineSecurity_No internet service': raw['OnlineSecurity'] == 'No internet service',
        'OnlineSecurity_Yes': raw['OnlineSecurity'] == 'Yes',
        'OnlineBackup_No internet service': raw['OnlineBackup'] == 'No internet service',
        'OnlineBackup_Yes': raw['OnlineBackup'] == 'Yes',
        'DeviceProtection_No internet service': raw['DeviceProtection'] == 'No internet service',
        'DeviceProtection_Yes': raw['DeviceProtection'] == 'Yes',
        'TechSupport_No internet service': raw['TechSupport'] == 'No internet service',
        'TechSupport_Yes': raw['TechSupport'] == 'Yes',
        'StreamingTV_No internet service': raw['StreamingTV'] == 'No internet service',
        'StreamingTV_Yes': raw['StreamingTV'] == 'Yes',
        'StreamingMovies_No internet service': raw['StreamingMovies'] == 'No internet service',
        'StreamingMovies_Yes': raw['StreamingMovies'] == 'Yes',
        'Contract_One year': raw['Contract'] == 'One year',
        'Contract_Two year': raw['Contract'] == 'Two year',
        'PaymentMethod_Credit card (automatic)': raw['PaymentMethod'] == 'Credit card (automatic)',
        'PaymentMethod_Electronic check': raw['PaymentMethod'] == 'Electronic check',
        'PaymentMethod_Mailed check': raw['PaymentMethod'] == 'Mailed check',
    }

    input_dict.update({k: int(v) for k, v in one_hot_map.items()})

    input_df = pd.DataFrame([input_dict])

    # Guarantee every expected column exists, in the exact training order
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }