#pip install git+https://github.com/openai/whisper.git torch

import whisper
import sounddevice as sd
import numpy as np
import wavio

model = whisper.load_model("tiny")

duration = 10
sample_rate = 16000

# Function to record audio
def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)

# Saving the speech file
def save_audio(audio, filename, sample_rate):
    wavio.write(filename, audio, sample_rate, sampwidth=2)
    print(f"Audio saved to {filename}")

# Transcribing the audio
def transcribe_audio(filename):

    print("Transcribing...")
    result = model.transcribe(filename)
    print("Transcription complete.")
    return result["text"]

def main():
    filename = "speech.wav"
    
    audio = record_audio(duration, sample_rate)
    save_audio(audio, filename, sample_rate)
    
    transcription = transcribe_audio(filename)
    #print("Transcription:", transcription)

if __name__ == "__main__":
    main()
