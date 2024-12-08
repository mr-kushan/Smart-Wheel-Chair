#### Smart Wheelchair for persons with disabilities in lower body and hearing impairements

The rising population of individuals with lower body disabilities and hearing impairments presents a challenge for current assistive devices. Traditional wheelchairs and aids lack advanced features like adaptive interfaces and real-time health monitoring, limiting user independence and increasing caregiver reliance. Innovative solutions are essential to enhance mobility, ensure safety, and provide personalized support for improved quality of life.

## Table of Contents
1. Functions
    - Function 1: ASL Detection & Speech Recognition
        - YOLOv5s Model ASL detection Model
        - Speech-to-Text Implementation
    
    - Function 3: Health Monitoring
        - Heart Attack Prediction Model
    - Function 4: Personalized Exercise Recommendation & Exercise Movement Tracking
        - Exercise Recommendation Model
        - Exercise Movement Tracking Model
2. API
3. How to Setup
4. Others


## 1. Functions

### Function 1: ASL Detection & Speech Recognition

#### Model 1: YOLOv5s ASL Detection

- **Dataset**: [`American Sign Language Letters`](https://public.roboflow.com/object-detection/american-sign-language-letters/1).
- **Model Code**:
    - `asl_fast_api.py`: API implementation for real-time ASL detection.
    - `Real_Time_ASL_Detection_Training.ipynb`: Notebook for training the ASL detection model.
    - `Real-Time ASL Detection - Testing.ipynb`: Notebook for testing the ASL detection model.
- **Model File**: `yolov5s.pt`
- **Technologies Used**: Python, YOLOv5, FastAPI
- **Model Description**: Real-time ASL detection model using YOLOv5 to identify American Sign Language gestures.
- **Inputs**: real-time camera footage.
- **Outputs**: Detected letter with confidence score.

### Model 2: Speech-to-Text using Whisper model

- **Code**: 
    - `speech2text.py`: Implementation of speech to text function for raspberry Pi using whisper model.
- **Technologies Used**: Python, whisper
- **Description**: Records speech and converts to text with openai whisper model.
- **Inputs**: Recorded speech file.
- **Outputs**: Transcribed text.


### Function 3: Health Monitoring

#### Model: Heart Attack Prediction

- **Dataset**: [`heart_attack_prediction_dataset`] from Kaggle 
- **Model Code**:
    - `Heart_Data_Generation.ipynb`: Data generation notebook for heart health monitoring.
    - `heart_attack_prediction_train.ipynb`: Training notebook for the heart attack prediction model.
- **Model File**: `heart_attack_prediction.pkl`
- **API**: `risk_prediction_fast_api.py`
- **Technologies Used**: Python, scikit-learn, FastAPI
- **Model Description**: Predicts the risk of heart attack based on health data.
- **Inputs**: Vital signs data.
- **Output**: Heart attack risk. (High/Low)


### Function 4: Personalized Exercise Recommendation

#### Model 1: Exercise Recommendation

- **Dataset**: [`wheelchair_neck_exercise_dataset`] from kaggle
- **Model Code**:
    - `Exercise_DS_Generation.ipynb`: Notebook for dataset generation for exercise recommendations.
    - `exercise_recommendation_train.ipynb`: Training notebook for the exercise recommendation model.
- **API**: `exercise_fast_api.py`
- **Technologies Used**: Python, scikit-learn, FastAPI, K- nearest neeighbors
- **Model Description**: Provides personalized exercise recommendations based on user data.
- **Inputs**: Age, Gender, Injury data.
- **Output**: Recommended neck exercises.


#### Model 2: Exercise Tracking

- **Dataset**: [`Exercise Image frames`] - own created data set

- **Model Code**:
    - `annonate.ipynb`: Notebook to annonate the keypoints in the images.
    - `Extract Frames.ipynb`: Notebook to extract frames from exercise videos.
    - `Augment.ipynb`: Notebook to augment the data frames.
    - `Train.ipynb`: Notebook to train the model.

- **Technologies Used**: Python, scikit-learn, OpenPose, FastAPI, CNN
- **Model Description**: Detects exercise movement in real-time.
- **Inputs**: Real-time camera footage.
- **Output**: Detected exercise with movement alert.

---

## 2. API

- **Technologies Used**: FastAPI, Swagger
- **API Files**:
    - `asl_fast_api.py`: API for ASL detection.
    - `risk_prediction_fast_api.py`: API for heart attack risk prediction.
    - `exercise_fast_api.py`: API for exercise recommendations.
    - `track_fast_api.py`: API for tracking exercises.

---

## 3. How to Setup

### Pre-requisites
- Python 3.8 or higher
- pip

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/mr-kushan/Smart-Wheel-Chair
    ```
2. Navigate to the project directory:
    ```bash
    cd mart-Wheel-Chair
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the APIs:
    ```bash
    uvicorn asl_fast_api:app --reload
    uvicorn risk_prediction_fast_api:app --reload
    uvicorn track_fast_api:app --reload
    uvicorn exercise_fast_api:app --reload
    ```

---