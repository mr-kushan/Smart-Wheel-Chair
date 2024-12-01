from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

with open("Function_3\Health_Monitoring\heart_attack_prediction.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI app
app = FastAPI()

# Input data model
class HealthData(BaseModel):
    heart_rate: float
    oxygen_level: float
    systolic_bp: float
    diastolic_bp: float
    body_temperature: float

# Prediction endpoint
@app.post("/predict")
def predict(data: HealthData):

    input_data = np.array([[data.heart_rate, data.oxygen_level, data.systolic_bp, 
                            data.diastolic_bp, data.body_temperature]])
    
    prediction = model.predict(input_data)
    
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    
    return {"heart_attack_risk": risk}
