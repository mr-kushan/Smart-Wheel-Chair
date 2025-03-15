#### Smart Wheelchair for persons with disabilities in lower body and hearing impairements

The rising population of individuals with lower body disabilities and hearing impairments presents a challenge for current assistive devices. Traditional wheelchairs and aids lack advanced features like adaptive interfaces and real-time health monitoring, limiting user independence and increasing caregiver reliance. Innovative solutions are essential to enhance mobility, ensure safety, and provide personalized support for improved quality of life.

## Table of Contents
1. Functions
    - Function 1: ASL Detection & Speech Recognition
        - YOLOv5s Model ASL detection Model
        - Speech-to-Text Implementation
    
    - Function 2: Pathfinding and Path Planning with Real-Time Monitoring
        - LIDAR-Based Navigation
        - Facial emotion detection for user status monitoring
    
    - Function 3: Health Monitoring
        - Heart Attack Prediction Model

    - Function 4: Personalized Exercise Recommendation & Exercise Movement Tracking
        - Personalized Exercise Recommendation Model
        - Exercise Movement Tracking Model
2. API
3. How to Setup
4. Others
---

### Function 1: Navigate wheelchair with American Sign Language Detection and Speech Recognition

## Overview

This function implements **ASL Detection & Speech Recognition** for the **Smart Wheelchair Rehabilitation System**. It features real-time American Sign Language (ASL) detection and a speech-to-text system to enhance accessibility and control for individuals with hearing impairments.

## Current implementations

1. **ASL Detection**: Detects American Sign Language gestures in real time using YOLOv5.
2. **Speech-to-Text**: Converts spoken language to text using OpenAI's Whisper model.

This function bridges communication gaps by enabling gesture-based and voice-based interactions.

## Features

- **Real-Time ASL Detection**: Identifies and interprets ASL gestures from video feeds.
- **Speech-to-Text Conversion**: Translates spoken words into text for seamless communication.
- **Lightweight and Efficient**: Designed for real-time performance on embedded systems like Raspberry Pi.

## Models

### ASL Detection
- **Description**: Detects and recognizes ASL gestures in real-time.
- **Dataset**: [`American Sign Language Letters Dataset`](https://public.roboflow.com/object-detection/american-sign-language-letters/1)
- **Trained**: using YOLOv5
- **Key Files**:
  - `asl_fast_api.py`: API implementation for ASL detection.
  - `Real_Time_ASL_Detection_Training.ipynb`: Notebook for training the YOLOv5 ASL detection model.
  - `Real-Time ASL Detection - Testing.ipynb`: Notebook for testing the trained ASL model.
- **Model File**: `yolov5s.pt` (pre-trained model weights).
- **Technologies Used**: Python, YOLOv5, FastAPI
- **Inputs**: Real-time camera footage.
- **Outputs**: Detected ASL gesture with confidence score.

### Speech-to-Text
- **Description**: Converts recorded speech into text using OpenAI's Whisper model.
- **Key File**:
  - `speech2text.py`: Script to implement the speech-to-text functionality.
- **Technologies Used**: Python, Whisper
- **Inputs**: Audio recording (speech file).
- **Outputs**: Transcribed text.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - YOLOv5 for ASL Detection
  - FastAPI for API implementation
  - OpenAI Whisper for speech-to-text conversion

---



### Function 2: Pathfinding and Path Planning with Real-Time Monitoring

This module integrates **LIDAR-based pathfinding with real-time monitoring to enhance navigation, safety, and accessibility for users.** It is designed for users with disabilities in the lower body and includes features for caregiver alerts and real-time path deviation detection.

## Features

1. **LIDAR-Based Navigation**  
   - Uses LIDAR sensors for obstacle detection and safe pathfinding.
   - Employs SLAM (Simultaneous Localization and Mapping) for dynamic map creation.
   - Integrated motor encoders for precise wheel movement tracking.

2. **Real-Time Monitoring**  
   - Facial emotion detection for user status monitoring.

## Technologies Used

- **Python**: Core programming language for implementation.
- **OpenCV**: Real-time facial recognition and image processing.
- **TensorFlow/Keras**: Model training for emotion recognition.
- **SLAM Toolbox**: For map creation and path planning.
- **Matplotlib**: Data visualization for debugging and analysis.

---



### Function 3: Integrated Health Monitoring System

This model implements **Integrated Health Monitoring System** for the **Smart Wheelchair Rehabilitation System**. It includes a heart attack risk prediction model designed to assess health risks based on vital signs data.

#### Model: Heart Attack Prediction

## Features

- **Heart Attack Risk Prediction**: Provides risk analysis based on vital health signs.
- **Real-Time Integration**: Designed for seamless integration into the wheelchair's monitoring system.
- **Scalable and Flexible**: Supports additional health monitoring functionalities.

## Model Details

### Heart Attack Prediction
- **Description**: Predicts the likelihood of a heart attack based on health data.
- **Dataset**: Heart attack prediction dataset from Kaggle.
- **Trained**: using XGBoost
- **Key Files**:
  - `Heart_Data_Generation.ipynb`: Notebook for preprocessing and augmenting health data.
  - `heart_attack_prediction_train.ipynb`: Notebook for training the heart attack prediction model.
  - `risk_prediction_fast_api.py`: FastAPI implementation for real-time risk prediction.
  - `heart_attack_prediction.pkl`: Trained model file.
- **Technologies Used**: Python, scikit-learn, FastAPI
- **Inputs**: Vital signs data (e.g., blood pressure, cholesterol levels).
- **Outputs**: Heart attack risk classification (High/Low).

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - scikit-learn for model development
  - FastAPI for API implementation

---



### Function 4: Adaptive Personalized Exercise Plan and Real-Time Motion Detection System with real-time Feedback

## Overview 

This function implements **Adaptive Personalized Exercise Plan and Real-Time Motion Detection System with real-time Feedback** for the **Smart Wheelchair Rehabilitation System**. It provides tailored neck exercise recommendations and tracks real-time exercise movements with feedback, focusing on aiding rehabilitation for individuals with hearing impairments.

## Current implementations

1. **Exercise Recommendation**: Generates personalized exercise plans based on user data.
2. **Exercise Tracking**: Monitors user exercises in real-time to ensure proper technique and progress.

The system integrates these functionalities into a cohesive rehabilitation tool, assisting users with tailored plans and actionable feedback.

## Features

- 1. **Personalized Exercise Recommendations**: Suggests neck exercises based on user characteristics such as age, gender, and injury details.
- 2. **Real-Time Motion Tracking**: Monitors exercise movements with real-time feedback.

## Models

## 1. Exercise Recommendation
- **Description**: Provides customized exercise plans for users.
- **Dataset**: [`Wheelchair Neck Exercise Dataset`] from Kaggle.
- **Trained**: using K nearest neighbors 
- **Key Files**:
  - `Exercise_DS_Generation.ipynb`: Notebook for dataset preprocessing and augmentation.
  - `exercise_recommendation_train.ipynb`: Model training notebook.
  - `exercise_fast_api.py`: API for real-time exercise recommendations.
- **Inputs**: User attributes such as age, gender, and injury details.
- **Outputs**: Recommended neck exercises.

## 2. Exercise Tracking
- **Description**: Detects and identifies exercise movements in real-time.
- **Dataset**: [`Exercise Image Frames`] - own created frames 
- **Trained**: using Convolutional neural networks (CNN) 
- **Key Files**:
  - `annotate.ipynb`: Annotates exercise keypoints in image data.
  - `Extract_Frames.ipynb`: Extracts image frames from exercise videos.
  - `Augment.ipynb`: Augments image frames for training.
  - `exercise_tracking_train.ipynb`: Trains the exercise tracking model.
  - `track_fast_api.py`: API for real-time exercise recommendations.
- **Inputs**: Real-time camera footage.
- **Outputs**: Detected exercises with movement alerts.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: 
  - scikit-learn
  - FastAPI
  - OpenPose

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