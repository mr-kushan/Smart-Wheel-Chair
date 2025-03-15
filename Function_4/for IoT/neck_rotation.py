import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def track_neck_rotation(landmarks, state, reps, last_change_time):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    left_angle = calculate_angle(nose, left_shoulder, [left_shoulder[0] - 0.1, left_shoulder[1]])
    right_angle = calculate_angle(nose, right_shoulder, [right_shoulder[0] + 0.1, right_shoulder[1]])
    symmetry = abs(left_angle - right_angle)

    threshold = 45
    cooldown = 0.5
    current_time = time.time()

    feedback = "Center head"
    if left_angle < threshold and current_time - last_change_time > cooldown:
        feedback = "Rotate right"
        if state == "right":
            reps += 1
            print(f"Rep {reps} completed")
        state = "left"
        last_change_time = current_time
    elif right_angle < threshold and current_time - last_change_time > cooldown:
        feedback = "Rotate left"
        if state == "left":
            reps += 1
            print(f"Rep {reps} completed")
        state = "right"
        last_change_time = current_time

    return reps, state, left_angle, right_angle, last_change_time, feedback

def save_to_csv(data, filename="records/real time data/neck_rotation_data.csv"):
    df = pd.DataFrame(data, columns=["Timestamp", "Reps", "Left Angle", "Right Angle", "Symmetry", "Feedback"])
    df.to_csv(filename, index=False)

def save_summary_to_records(exercise_name, total_reps, total_time, avg_symmetry, avg_rotation_angle, performance, recommended_reps, filename="records/final performance data/neck_rotation.csv"):
    summary_data = {"Exercise Name": [exercise_name], "Total Reps": [total_reps], "Total Time (s)": [total_time], "Average Symmetry": [avg_symmetry], "Average Rotation Angle": [avg_rotation_angle], "Performance": [performance], "Recommended Reps": [recommended_reps]}
    df = pd.concat([pd.read_csv(filename), pd.DataFrame(summary_data)], ignore_index=True) if pd.io.common.file_exists(filename) else pd.DataFrame(summary_data)
    df.to_csv(filename, index=False)

def evaluate_performance(avg_symmetry, avg_rotation_angle, total_reps, recommended_reps):
    symmetry_good = avg_symmetry < 10
    angle_good = avg_rotation_angle < 60
    reps_factor = total_reps / recommended_reps if recommended_reps > 0 else 1.0
    if symmetry_good and angle_good:
        if reps_factor >= 1.0: return "Excellent"
        elif reps_factor >= 0.75: return "Good"
        else: return "Average"
    elif avg_symmetry < 20 and avg_rotation_angle < 80:
        if reps_factor >= 1.0: return "Good"
        elif reps_factor >= 0.5: return "Average"
        else: return "Poor"
    return "Poor"

def neck_rotation_tracker(stop_event, recommended_reps=10):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    reps, state, tracked_data, last_change_time = 0, "neutral", [], start_time
    feedback = "Starting..."

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            reps, state, left_angle, right_angle, last_change_time, feedback = track_neck_rotation(landmarks, state, reps, last_change_time)
            timestamp = time.time() - start_time
            tracked_data.append([timestamp, reps, left_angle, right_angle, abs(left_angle - right_angle), feedback])

            current_time = int(timestamp)
            cv2.putText(frame, f"Time: {current_time}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Reps: {reps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Target: {recommended_reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Feedback: {feedback}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Neck Rotation Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    save_to_csv(tracked_data)
    total_time = time.time() - start_time
    avg_symmetry = np.mean([row[4] for row in tracked_data]) if tracked_data else 0
    avg_rotation_angle = np.mean([(row[2] + row[3]) / 2 for row in tracked_data]) if tracked_data else 0
    performance = evaluate_performance(avg_symmetry, avg_rotation_angle, reps, recommended_reps)
    save_summary_to_records("Neck Rotation", reps, total_time, avg_symmetry, avg_rotation_angle, performance, recommended_reps)
    print(f"Performance: {performance}, Reps: {reps}/{recommended_reps}")