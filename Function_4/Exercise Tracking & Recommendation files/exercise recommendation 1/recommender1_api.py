from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Recommender class
class Recommender:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.gender_mapping = {"Male": 1, "Female": 0}
        self.injury_mapping = {
            "Quadriplegia": 0,
            "Multiple Sclerosis": 1,
            "Paraplegia": 2,
            "Spinal Cord Injury": 3,
            "Amputation": 4
        }
        self.fitness_mapping = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        self.rep_counts = {"Beginner": 6, "Intermediate": 10, "Advanced": 15}

    def preprocess_data(self, data):
        """Preprocess exercise_dataset.csv or user input."""
        data = data.copy()
        data["Gender"] = data["Gender"].map(self.gender_mapping)
        data["Injury"] = data["Injury"].map(self.injury_mapping)
        data["Fitness Level"] = data["Fitness Level"].map(self.fitness_mapping)
        data[["Age"]] = self.scaler.transform(data[["Age"]])
        return data

    def fit(self, data):
        """Fit the scaler on the exercise dataset."""
        data = data.copy()
        self.scaler.fit(data[["Age"]])
        return self

    def create_user_profile(self, user_age, user_gender, user_injury, user_fitness_level):
        """Create a feature vector for the user."""
        user_data = pd.DataFrame({
            "Age": [user_age],
            "Gender": [self.gender_mapping.get(user_gender, 0)],
            "Injury": [self.injury_mapping.get(user_injury, 0)],
            "Fitness Level": [self.fitness_mapping.get(user_fitness_level, 1)]
        })
        user_data[["Age"]] = self.scaler.transform(user_data[["Age"]])
        return user_data.values.reshape(1, -1)

    def recommend_exercises(self, data, user_profile, fitness_level, top_n=5):
        """Recommend exercises based on cosine similarity."""
        exercise_vectors = data[["Age", "Gender", "Injury", "Fitness Level"]].values
        similarities = cosine_similarity(user_profile, exercise_vectors).flatten()
        data["Similarity"] = similarities
        exercise_summary = (
            data.groupby("Exercise Name")
            .agg({"Similarity": "mean"})
            .reset_index()
        )
        recommendations = exercise_summary.sort_values(by="Similarity", ascending=False).head(top_n)
        recommended_reps = self.rep_counts.get(fitness_level, 6)
        return [{"Exercise Name": row["Exercise Name"], "Recommended Reps": recommended_reps}
                for _, row in recommendations.iterrows()]


with open("models/recommender1.pkl", "rb") as f:
    recommender = pickle.load(f)


df = pd.read_csv("data/exercise_dataset.csv")
processed_df = recommender.preprocess_data(df)


# FastAPI app
app = FastAPI()

# Input model
class UserProfile(BaseModel):
    age: int
    gender: str
    injury: str
    fitness_level: str

# API endpoint
@app.post("/recommend")
async def get_recommendations(user: UserProfile):
    try:
        user_profile = recommender.create_user_profile(user.age, user.gender, user.injury, user.fitness_level)
        recommendations = recommender.recommend_exercises(processed_df, user_profile, user.fitness_level)
        return {"recommended_exercises": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Exercise Recommendation 1 API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)