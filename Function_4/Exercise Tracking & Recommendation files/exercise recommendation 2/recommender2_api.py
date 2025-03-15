from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ExerciseRecommender class
class ExerciseRecommender:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.difficulty_mapping = {"Easy": 1, "Medium": 2, "Hard": 3}
        self.performance_mapping = {"Good": 3, "Average": 2, "Bad": 1}

    def preprocess_data(self, data):
        data = data.copy()
        data["Difficulty Level"] = data["Difficulty Level"].map(self.difficulty_mapping)
        data["Performance Score"] = data["Overall Performance"].map(self.performance_mapping)
        data[["Total Reps", "Total Time (s)"]] = self.scaler.transform(data[["Total Reps", "Total Time (s)"]])
        return data

    def fit(self, data):
        data = data.copy()
        self.scaler.fit(data[["Total Reps", "Total Time (s)"]])
        return self

    def create_user_profile(self, data, user_id):
        user_data = data[data["User ID"] == user_id]
        user_profile = user_data[["Total Reps", "Total Time (s)", "Difficulty Level", "Performance Score"]].mean()
        return user_profile.values.reshape(1, -1)

    def recommend_exercises(self, data, user_profile, top_n=5):
        exercise_vectors = data[["Total Reps", "Total Time (s)", "Difficulty Level", "Performance Score"]].values
        similarities = cosine_similarity(user_profile, exercise_vectors).flatten()
        data["Similarity"] = similarities
        exercise_summary = (
            data.groupby("Exercise Name")
            .agg({
                "Similarity": "mean",
                "Total Reps": "mean",
                "Difficulty Level": "mean",
                "Performance Score": "mean",
                "Total Time (s)": "mean"
            })
            .reset_index()
        )
        exercise_summary["Adjusted Similarity"] = (
            exercise_summary["Similarity"] * (1 + (3 - exercise_summary["Performance Score"]) * 0.2)
        )
        recommendations = exercise_summary.sort_values(by="Adjusted Similarity", ascending=False).head(top_n)
        return recommendations[["Exercise Name"]]

with open("models/recommender2.pkl", "rb") as f:
    recommender = pickle.load(f)

# Static rep counts based on inferred performance level
rep_counts = {
    "Beginner": 6,
    "Intermediate": 10,
    "Advanced": 15
}

# Difficulty and gender mappings
difficulty_map = {
    "Neck Side Stretch": "Easy",
    "Neck Rotation": "Easy",
    "Chin Tucks": "Medium",
    "Shoulder Shrugs": "Medium",
    "Neck Flexion": "Easy",
    "Seated Head Turns": "Medium",
    "Neck Side Bend": "Medium",
    "Neck Isometrics": "Medium",
    "Neck Extension": "Easy"
}
gender_mapping = {1: "Female", 2: "Male"}

input_folder = "records/final performance data"

# FastAPI app
app = FastAPI()

# Input model
class UserInput(BaseModel):
    user_id: int

# Function to calculate overall performance
def calculate_overall_performance(performance_column):
    performance_counts = Counter(performance_column)
    return performance_counts.most_common(1)[0][0]

# Process a single exercise file
def process_exercise_file(filepath: str, user_id: int):
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        return None

    total_time = data["Total Time (s)"].sum()
    total_reps = data["Total Reps"].sum()
    overall_performance = calculate_overall_performance(data["Performance"])

    exercise_name = os.path.splitext(os.path.basename(filepath))[0]
    difficulty_level = difficulty_map.get(exercise_name, "Unknown")
    gender = gender_mapping.get(user_id, "Unknown")

    return {
        "User ID": user_id,
        "Gender": gender,
        "Exercise Name": exercise_name,
        "Difficulty Level": difficulty_level,
        "Total Time (s)": total_time,
        "Total Reps": total_reps,
        "Overall Performance": overall_performance
    }

# Consolidate performance data from server folder
def consolidate_user_performance(user_id: int):
    exercise_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".csv")]
    consolidated_data = []
    for file in exercise_files:
        exercise_data = process_exercise_file(file, user_id)
        if exercise_data:
            consolidated_data.append(exercise_data)
    return pd.DataFrame(consolidated_data)

# Recommendation function
def recommend_exercises(user_id: int, csv_data: pd.DataFrame, top_n=5):
    processed_data = recommender.preprocess_data(csv_data)
    if user_id not in processed_data["User ID"].values:
        raise ValueError(f"User ID {user_id} not found in the data")
    
    user_profile = recommender.create_user_profile(processed_data, user_id)
    recommendations = recommender.recommend_exercises(processed_data, user_profile, top_n=top_n)
    
    avg_performance = processed_data[processed_data["User ID"] == user_id]["Performance Score"].mean()
    performance_level = "Advanced" if avg_performance >= 2.5 else "Intermediate" if avg_performance >= 1.5 else "Beginner"
    recommended_reps = rep_counts.get(performance_level, 6)
    
    return [{"Exercise Name": row["Exercise Name"], "Recommended Reps": recommended_reps} 
            for _, row in recommendations.iterrows()]

# Combined merge and recommend endpoint
@app.post("/recommend")
async def get_recommendations(user: UserInput):
    try:
        consolidated_data = consolidate_user_performance(user.user_id)
        if consolidated_data.empty:
            raise ValueError("No performance data found for this user in the server folder")
        
        recommendations = recommend_exercises(user.user_id, consolidated_data)
        return {"recommended_exercises": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Exercise Recommendation 2 API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)