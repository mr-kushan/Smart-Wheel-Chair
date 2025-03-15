from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import mediapipe as mp
import numpy as np
import cv2
import pickle

# IFastAPI app
app = FastAPI()

# Load ASL detection model
model_dict = pickle.load(open('Function 1//ASL Detection 2//asl_detection_model.p', 'rb'))
model = model_dict['model']

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8)

@app.get("/predict")
def predict():
    try:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally to simulate a mirror effect (for a realistic experience)
            frame = cv2.flip(frame, 1)

            H, W, _ = frame.shape

            # Converting to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=5, circle_radius=10),
                        mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=5, circle_radius=10)
                    )

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

                if len(data_aux) >= 42:
                    x1 = max(int(min(x_) * W) - 10, 0)
                    y1 = max(int(min(y_) * H) - 10, 0)
                    x2 = min(int(max(x_) * W) + 10, W)
                    y2 = min(int(max(y_) * H) + 10, H)

                    prediction = model.predict([np.array(data_aux)[0:42]])[0]

                    cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (255, 99, 173), -1)
                    cv2.putText(frame, prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('ASL Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return JSONResponse(content={"message": "Webcam test completed. Press 'q' to exit."}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)