from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np


with open("Function_4/Personalized_Exercise_Recommendation/models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("Function_4/Personalized_Exercise_Recommendation/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Function_4/Personalized_Exercise_Recommendation/models/recommender_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

df = pd.read_csv('Function_4/Personalized_Exercise_Recommendation/wheelchair_neck_exercise_dataset.csv')

# FastAPI app
app = FastAPI()

class UserProfile(BaseModel):
    age: int
    gender: str
    injury: str

def recommend_exercises(user_age, user_gender, user_injury, top_n=5):

    user_data = pd.DataFrame([[user_gender, user_injury]], columns=['Gender', 'Injury'])
    categorical_encoded = encoder.transform(user_data).toarray()
    age_scaled = scaler.transform([[user_age]])
    user_features = np.hstack((age_scaled, categorical_encoded))
    
    distances, indices = knn_model.kneighbors(user_features, n_neighbors=top_n)
    
    recommended_exercises = df.iloc[indices[0]]['Exercise Name'].values
    return recommended_exercises

# API endpoint
@app.post("/recommend")
async def get_recommendations(user: UserProfile):

    recommendations = recommend_exercises(user.age, user.gender, user.injury)
    return {"recommended_exercises": recommendations.tolist()}
