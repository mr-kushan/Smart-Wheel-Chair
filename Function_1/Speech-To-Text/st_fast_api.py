import whisper
from fastapi import FastAPI, File, UploadFile
import uvicorn
import tempfile
import os


model = whisper.load_model("tiny")

# FastAPI app
app = FastAPI()

# Endpoint to upload and transcribe audio
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        text = result["text"]
    finally:
        os.remove(tmp_path)

    return {"Text": text}
