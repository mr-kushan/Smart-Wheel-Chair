import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def track_shoulder_shrugs(landmarks, state, reps, last_change_time):
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    chest_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2

    shoulder_height = (left_shoulder_y + right_shoulder_y) / 2 - chest_y 
    symmetry = abs(left_shoulder_y - right_shoulder_y)

    threshold_up = -0.1 
    threshold_down = -0.05  
    debounce_time = 0.5 
    current_time = time.time()

    feedback = "Relax shoulders"
    if shoulder_height < threshold_up and current_time - last_change_time > debounce_time:
        feedback = "Shrug up"
        if state == "down":
            reps += 1
            print(f"Rep {reps} completed")
        state = "up"
        last_change_time = current_time
    elif shoulder_height > threshold_down and current_time - last_change_time > debounce_time:
        feedback = "Relax down"
        if state == "up":
            reps += 1
            print(f"Rep {reps} completed")
        state = "down"
        last_change_time = current_time

    return reps, state, shoulder_height, symmetry, last_change_time, feedback

def save_to_csv(data, filename="records/real time data/shoulder_shrugs_data.csv"):
    df = pd.DataFrame(data, columns=["Timestamp", "Reps", "Shoulder Height", "Symmetry", "Feedback"])
    df.to_csv(filename, index=False)

def save_summary_to_records(exercise_name, total_reps, total_time, avg_symmetry, avg_height, performance, recommended_reps, filename="records/final performance data/shoulder_shrugs.csv"):
    summary_data = {
        "Exercise Name": [exercise_name],
        "Total Reps": [total_reps],
        "Total Time (s)": [total_time],
        "Average Symmetry": [avg_symmetry],
        "Average Height": [avg_height],
        "Performance": [performance],
        "Recommended Reps": [recommended_reps]
    }
    df = pd.concat([pd.read_csv(filename), pd.DataFrame(summary_data)], ignore_index=True) if pd.io.common.file_exists(filename) else pd.DataFrame(summary_data)
    df.to_csv(filename, index=False)

def evaluate_performance(avg_symmetry, avg_height, total_reps, recommended_reps):
    symmetry_good = avg_symmetry < 0.05  # Shoulders move evenly
    height_good = avg_height < -0.07  # Average height indicates sufficient shrug
    reps_factor = total_reps / recommended_reps if recommended_reps > 0 else 1.0
    if symmetry_good and height_good:
        if reps_factor >= 1.0: return "Excellent"
        elif reps_factor >= 0.75: return "Good"
        else: return "Average"
    elif reps_factor >= 0.5: return "Average"
    return "Poor"

def shoulder_shrugs_tracker(stop_event, recommended_reps=10):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    reps, state, tracked_data, last_change_time = 0, "neutral", [], start_time
    feedback = "Starting..."

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            reps, state, shoulder_height, symmetry, last_change_time, feedback = track_shoulder_shrugs(landmarks, state, reps, last_change_time)
            timestamp = time.time() - start_time
            tracked_data.append([timestamp, reps, shoulder_height, symmetry, feedback])

            current_time = int(timestamp)
            cv2.putText(frame, f"Time: {current_time}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Reps: {reps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Target: {recommended_reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Feedback: {feedback}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Shoulder Shrugs Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_to_csv(tracked_data)
    total_time = time.time() - start_time
    avg_symmetry = np.mean([row[3] for row in tracked_data]) if tracked_data else 0
    avg_height = np.mean([row[2] for row in tracked_data]) if tracked_data else 0
    performance = evaluate_performance(avg_symmetry, avg_height, reps, recommended_reps)
    save_summary_to_records("Shoulder Shrugs", reps, total_time, avg_symmetry, avg_height, performance, recommended_reps)
    print(f"Performance: {performance}, Reps: {reps}/{recommended_reps}")
