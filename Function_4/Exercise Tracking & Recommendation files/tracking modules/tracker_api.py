from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
from typing import List
import uvicorn
from chin_tucks import chin_tucks_tracker
from head_turns import head_turns_tracker 
from neck_extension import neck_extension_tracker
from neck_flexion import neck_flexion_tracker
from neck_isometrics import neck_isometrics_tracker
from neck_rotation import neck_rotation_tracker
from neck_side_bend import neck_side_bend_tracker
from neck_side_stretch import neck_side_stretch_tracker
from shoulder_shrugs import shoulder_shrugs_tracker

app = FastAPI()

EXERCISE_TRACKERS = {
    "chin_tucks": chin_tucks_tracker,
    "head_turns": head_turns_tracker,
    "neck_extension": neck_extension_tracker,
    "neck_flexion": neck_flexion_tracker,
    "neck_isometrics": neck_isometrics_tracker,
    "neck_rotation": neck_rotation_tracker,
    "neck_side_bend": neck_side_bend_tracker,
    "neck_side_stretch": neck_side_stretch_tracker,
    "shoulder_shrugs": shoulder_shrugs_tracker
}

recommended_exercises = []
current_tracker_thread = None
stop_event = threading.Event()

class StartExerciseRequest(BaseModel):
    recommended_reps: int

@app.post("/exercises")
async def set_exercises(exercises: List[str]):
    """Set the 5 recommended exercises from the frontend."""
    global recommended_exercises
    if len(exercises) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 exercises must be provided")
    invalid_exercises = [ex for ex in exercises if ex not in EXERCISE_TRACKERS]
    if invalid_exercises:
        raise HTTPException(status_code=400, detail=f"Invalid exercises: {invalid_exercises}")
    recommended_exercises = exercises
    return {"message": "Recommended exercises set", "exercises": recommended_exercises}

@app.get("/exercises")
async def get_exercises():
    """Retrieve the current list of recommended exercises."""
    return {"exercises": recommended_exercises}

@app.post("/start/{exercise_name}")
async def start_exercise(exercise_name: str, request: StartExerciseRequest):
    """Start the tracking module for the specified exercise with recommended reps."""
    global current_tracker_thread, stop_event
    if exercise_name not in recommended_exercises:
        raise HTTPException(status_code=400, detail="Exercise not in recommended list")
    if current_tracker_thread and current_tracker_thread.is_alive():
        raise HTTPException(status_code=400, detail="Another exercise is currently running")

    stop_event.clear()
    tracker_func = EXERCISE_TRACKERS[exercise_name]
    current_tracker_thread = threading.Thread(
        target=tracker_func, 
        args=(stop_event, request.recommended_reps)
    )
    current_tracker_thread.start()
    return {"message": f"Started tracking {exercise_name}"}

@app.post("/stop")
async def stop_exercise():
    """Stop the currently running exercise."""
    global current_tracker_thread, stop_event
    if not current_tracker_thread or not current_tracker_thread.is_alive():
        raise HTTPException(status_code=400, detail="No exercise is currently running")
    
    stop_event.set()
    current_tracker_thread.join()
    current_tracker_thread = None
    return {"message": "Exercise stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)