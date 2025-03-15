import pandas as pd
import os
from collections import Counter


input_folder = "final_performance_data"
output_file = "user_progress_data.csv"


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


def calculate_overall_performance(performance_column):
    performance_counts = Counter(performance_column)
    return performance_counts.most_common(1)[0][0]


def process_exercise_file(filepath, user_id):

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


def consolidate_user_performance(user_id):

    exercise_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".csv")]

    consolidated_data = []
    for file in exercise_files:
        exercise_data = process_exercise_file(file, user_id)
        if exercise_data:
            consolidated_data.append(exercise_data)

    return pd.DataFrame(consolidated_data)


def generate_user_performance_data(user_id=1):

    user_performance_data = consolidate_user_performance(user_id)

    user_performance_data.to_csv(output_file, index=False)


generate_user_performance_data(user_id=1)
