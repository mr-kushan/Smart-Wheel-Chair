from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import cv2
import asyncio
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='Function_1\\ASL_Detection\\yolov5\\runs\\train\\exp\\weights\\best.pt', force_reload=True)


# WebSocket endpoint for real-time video feed
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            predictions = results.pandas().xyxy[0].to_dict(orient="records")

            await websocket.send_json(predictions)

            for pred in predictions:
                cv2.rectangle(frame, (int(pred['xmin']), int(pred['ymin'])),
                              (int(pred['xmax']), int(pred['ymax'])),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"{pred['name']} {pred['confidence']:.2f}", 
                            (int(pred['xmin']), int(pred['ymin']) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("ASL Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        await websocket.close()
