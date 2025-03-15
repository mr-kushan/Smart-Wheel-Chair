from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

with open(r"E:\SLIIT\Final Research project\Smart-Wheel-Chair-ML\Function 3\Health Monitoring\new\heart_attack_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI app
app = FastAPI()

# Input data model
class HealthData(BaseModel):
    heart_rate: float
    oxygen_level: float
    body_temperature: float

# Prediction endpoint
@app.post("/predict")
def predict(data: HealthData):

    input_data = np.array([[data.heart_rate, data.oxygen_level, data.body_temperature]])
    
    prediction = model.predict(input_data)
    
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    
    return {"heart_attack_risk": risk}
