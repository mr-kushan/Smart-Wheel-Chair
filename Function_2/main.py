import os
import sys
import traceback

sys.path.insert(0, os.getcwd())

from function_02.Emotion_Recognition.live_emotion import detect_emotion


# ----- F2: Emotion Detection -----
def emotion_detection(frame):
    try:
        emotion_label = detect_emotion(frame)
        print(emotion_label)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    

if __name__ == "__main__":
    print("Running") 