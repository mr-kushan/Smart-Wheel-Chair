import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def track_chin_tucks(landmarks, state, reps, last_change_time):
    nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
    avg_shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2
    displacement = nose_x - avg_shoulder_x
    symmetry = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)

    threshold_forward = 0.1
    threshold_backward = -0.1
    cooldown = 0.5
    current_time = time.time()

    feedback = "Hold position"
    if displacement > threshold_forward and current_time - last_change_time > cooldown:
        feedback = "Pull chin back"
        if state == "backward":
            reps += 1
            print(f"Rep {reps} completed")
        state = "forward"
        last_change_time = current_time
    elif displacement < threshold_backward and current_time - last_change_time > cooldown:
        feedback = "Push chin forward"
        if state == "forward":
            reps += 1
            print(f"Rep {reps} completed")
        state = "backward"
        last_change_time = current_time

    return reps, state, displacement, symmetry, last_change_time, feedback

def save_to_csv(data, filename="records/real time data/chin_tucks_data.csv"):
    df = pd.DataFrame(data, columns=["Timestamp", "Reps", "Displacement", "Symmetry", "Feedback"])
    df.to_csv(filename, index=False)

def save_summary_to_records(exercise_name, total_reps, total_time, avg_symmetry, avg_displacement, performance, recommended_reps, filename="records/final performance data/chin_tucks.csv"):
    summary_data = {"Exercise Name": [exercise_name], "Total Reps": [total_reps], "Total Time (s)": [total_time], "Average Symmetry": [avg_symmetry], "Average Displacement": [avg_displacement], "Performance": [performance], "Recommended Reps": [recommended_reps]}
    df = pd.concat([pd.read_csv(filename), pd.DataFrame(summary_data)], ignore_index=True) if pd.io.common.file_exists(filename) else pd.DataFrame(summary_data)
    df.to_csv(filename, index=False)

def evaluate_performance(avg_symmetry, avg_displacement, total_reps, recommended_reps):
    symmetry_good = avg_symmetry < 0.02
    displacement_good = abs(avg_displacement) > 0.05
    reps_factor = total_reps / recommended_reps if recommended_reps > 0 else 1.0
    if symmetry_good and displacement_good:
        if reps_factor >= 1.0: return "Excellent"
        elif reps_factor >= 0.75: return "Good"
        else: return "Average"
    elif avg_symmetry < 0.05 and abs(avg_displacement) > 0.03:
        if reps_factor >= 1.0: return "Good"
        elif reps_factor >= 0.5: return "Average"
        else: return "Poor"
    return "Poor"

def chin_tucks_tracker(stop_event, recommended_reps=10):
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
            reps, state, displacement, symmetry, last_change_time, feedback = track_chin_tucks(landmarks, state, reps, last_change_time)
            timestamp = time.time() - start_time
            tracked_data.append([timestamp, reps, displacement, symmetry, feedback])

            current_time = int(timestamp)
            cv2.putText(frame, f"Time: {current_time}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Reps: {reps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Target: {recommended_reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Feedback: {feedback}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Chin Tucks Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    save_to_csv(tracked_data)
    total_time = time.time() - start_time
    avg_symmetry = np.mean([row[3] for row in tracked_data]) if tracked_data else 0
    avg_displacement = np.mean([row[2] for row in tracked_data]) if tracked_data else 0
    performance = evaluate_performance(avg_symmetry, avg_displacement, reps, recommended_reps)
    save_summary_to_records("Chin Tucks", reps, total_time, avg_symmetry, avg_displacement, performance, recommended_reps)
    print(f"Performance: {performance}, Reps: {reps}/{recommended_reps}")