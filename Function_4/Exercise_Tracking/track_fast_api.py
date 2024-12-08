import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from typing import List

# FastAPI app
app = FastAPI()

model = load_model("Function_4/Exercise_Tracking/tracking_model.h5", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
num_classes = 6 
frame_sequence = []
sequence_length = 5


def preprocess_frame(frame):

    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return normalized_frame


@app.get("/")
def read_root():
    return {"message": "Real-time exercise tracking"}


#Prediction endpoint
@app.get("/predict-webcam")
def predict_webcam():
    global frame_sequence

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Unable to access webcam"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        frame_sequence.append(processed_frame)

        if len(frame_sequence) > sequence_length:
            frame_sequence.pop(0)

        if len(frame_sequence) == sequence_length:

            input_sequence = np.expand_dims(frame_sequence, axis=0)
            predictions = model.predict(input_sequence)
            predicted_class = np.argmax(predictions)

            class_label = f"Predicted Exercise: {predicted_class}"
            cv2.putText(frame, class_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"message": "Webcam feed closed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)