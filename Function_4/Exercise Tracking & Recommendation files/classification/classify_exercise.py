import cv2
import mediapipe as mp
import numpy as np
import joblib
import itertools
import pandas as pd

# Load the trained model
model_path = "exercise_classifier.pkl"
rf_model = joblib.load(model_path)

# Load normalization factors
combined_training_csv = "combined_features.csv"
training_data = pd.read_csv(combined_training_csv)

# Compute normalization factors from training data
normalization_factors = {
    col: training_data[col].max() if col.startswith("distance") or col.startswith("angle") else 1
    for col in training_data.columns if col != "class" and col != "frame"
}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define the relevant keypoints
relevant_keypoints = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
]

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# Function to calculate angle between three points
def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Extract features (distances and angles)
def extract_features(keypoints):
    features = []

    # Compute pairwise distances
    for i, j in itertools.combinations(range(len(keypoints)), 2):
        point1 = keypoints[i]
        point2 = keypoints[j]
        if None not in point1 and None not in point2:
            features.append(calculate_distance(point1, point2))
        else:
            features.append(0)  # Handle missing values

    # Compute angles for keypoint triplets
    for i, j, k in itertools.combinations(range(len(keypoints)), 3):
        point1 = keypoints[i]
        point2 = keypoints[j]
        point3 = keypoints[k]
        if None not in point1 and None not in point2 and None not in point3:
            features.append(calculate_angle(point1, point2, point3))
        else:
            features.append(0)  # Handle missing values

    return features

# Normalize features using the saved normalization factors
def normalize_features(features):
    normalized_features = [
        features[i] / normalization_factors.get(f"feature_{i}", 1)
        for i in range(len(features))
    ]
    return normalized_features

# Real-time exercise detection with feature logging
def real_time_exercise_detection(frame_rate=10, output_csv="real_time_features.csv"):
    cap = cv2.VideoCapture(0)  # Open the webcam
    interval = 1 / frame_rate  # Time interval between frames
    last_time = 0

    # Initialize list for storing real-time features
    logged_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame at the specified frame rate
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - last_time >= interval:
            last_time = current_time

            # Convert frame to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract keypoints
                keypoints = [
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ]
                keypoints = [keypoints[kp.value] for kp in relevant_keypoints]

                # Extract features
                features = extract_features(keypoints)

                # Normalize features
                normalized_features = normalize_features(features)

                # Log normalized features with frame number
                if len(normalized_features) == len(normalization_factors):
                    frame_data = [int(last_time * 1000)] + normalized_features  # Add timestamp as frame number
                    logged_features.append(frame_data)

                    # Predict exercise class
                    prediction = rf_model.predict([normalized_features])[0]

                    # Overlay prediction on the frame
                    cv2.putText(frame, f"Exercise: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Exercise Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save logged features to a CSV file
    if logged_features:
        columns = ["frame"] + [f"feature_{i}" for i in range(len(logged_features[0]) - 1)]
        logged_features_df = pd.DataFrame(logged_features, columns=columns)
        logged_features_df.to_csv(output_csv, index=False)
        print(f"Real-time features saved to {output_csv}")

# Run the real-time detection with logging
real_time_exercise_detection(frame_rate=10)
