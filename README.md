# Traffic-Prediction-System-YOLOv8-based-

This project is a traffic video prediction system that analyzes traffic videos to predict congestion and traffic patterns using advanced machine learning algorithms. The system uses a fine-tuned YOLOv8 model to detect vehicles and determine traffic intensity in different lanes. Additionally, it performs weather prediction based on video data.

## Features
- Vehicle Detection: The system detects vehicles in different lanes and counts them.
- Traffic Intensity Prediction: It classifies traffic conditions as either "Heavy" or "Smooth" based on the vehicle count in each lane.
- Weather Detection: Predicts the weather conditions (Rainy, Foggy, Clear Sky, etc.) based on visual analysis of the traffic video.
- Web Interface: Users can upload traffic videos through a simple web interface, and the system provides real-time analysis results.
- Flask Backend: The backend processes the uploaded video, runs the YOLOv8 model for vehicle detection, and returns the prediction results.


## Tech Stack
- Backend: Python, Flask, OpenCV, YOLOv8 (Ultralytics)
- Frontend: HTML, CSS (Bootstrap), JavaScript
- Machine Learning Model: YOLOv8 (Pretrained and fine-tuned)


## Project Structure

.
├── models
│   └── best.pt                # Pretrained YOLOv8 model
├── static
│   └── images
│       └── bg.jpg             # Background image for the web page
├── templates
│   └── web.html               # Frontend HTML file
├── sample_video.mp4           # Sample traffic video for testing
├── processed_sample_video.avi # Processed output video with predictions
├── real_time_traffic_analysis.py # Flask backend for real-time traffic analysis
├── README.md                  # Project documentation
└── requirements.txt           # Required Python dependencies


## How It Works
- Video Upload: The user uploads a traffic video via the web interface.
- Video Processing: The YOLOv8 model processes each frame of the video to detect vehicles in two lanes.
- Traffic Prediction: The system counts vehicles in both lanes and classifies traffic as either "Heavy" or "Smooth".
- Weather Detection: The system predicts weather conditions by analyzing brightness, motion intensity, and other visual clues in the video.
- Results Display: The traffic intensity and weather conditions are displayed to the user after the analysis is complete.
Installation
Prerequisites
Python 3.x
YOLOv8 (Ultralytics) model file (best.pt)
Step-by-Step Setup


## Clone the repository:
git clone https://github.com/yourusername/traffic-prediction-system.git
cd traffic-prediction-system

## Install the dependencies:
Install the required Python packages using requirements.txt:
pip install -r requirements.txt

## The key dependencies include:
Flask
OpenCV
Ultralytics YOLO

## Run the Flask application:
python real_time_traffic_analysis.py
Access the web interface:

**Open your browser and go to http://127.0.0.1:5000 to upload a traffic video for prediction.**

## Usage
Navigate to the web page.
Upload a traffic video file (e.g., .mp4).
Click the "Upload & Predict" button.
Wait for the results to be displayed, showing:
Vehicle count and traffic intensity for both left and right lanes.
Predicted weather condition based on video analysis.
YOLOv8 Model
The YOLOv8 model used in this project is pretrained and fine-tuned on traffic-specific datasets. It can detect vehicles and other objects in the traffic scene with high accuracy. The model file best.pt is required for the system to work.


