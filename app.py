from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask("real_time_traffic_analysis.py")

# Load the YOLO model
best_model = YOLO('models/best.pt')

# Threshold for heavy traffic
heavy_traffic_threshold = 10

# Vertices for quadrilaterals
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

x1, x2 = 325, 635
lane_threshold = 609

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
background_color = (0, 0, 255)



@app.route('/')
def index():
    return render_template('web.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains a file part
    if 'videoFile' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    # Get the file from the request
    file = request.files['videoFile']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    video_path = 'sample_video.mp4'
    file.save(video_path)
    
    # Open the video and process it
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    left_lane_count = 0
    right_lane_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection_frame = frame.copy()
        detection_frame[:x1, :] = 0
        detection_frame[x2:, :] = 0

        results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
        bounding_boxes = results[0].boxes

        vehicles_in_left_lane = 0
        vehicles_in_right_lane = 0

        for box in bounding_boxes.xyxy:
            if box[0] < lane_threshold:
                vehicles_in_left_lane += 1
            else:
                vehicles_in_right_lane += 1

        left_lane_count += vehicles_in_left_lane
        right_lane_count += vehicles_in_right_lane
        total_frames += 1

    # Close video capture
    cap.release()

    # Calculate average vehicles per frame
    avg_vehicles_left = left_lane_count / total_frames if total_frames else 0
    avg_vehicles_right = right_lane_count / total_frames if total_frames else 0

    traffic_intensity_left = "Heavy" if avg_vehicles_left > heavy_traffic_threshold else "Smooth"
    traffic_intensity_right = "Heavy" if avg_vehicles_right > heavy_traffic_threshold else "Smooth"

    # Return the result in JSON format
    return jsonify({
        'left_lane': f'Vehicles in Left Lane: {avg_vehicles_left}',
        'right_lane': f'Vehicles in Right Lane: {avg_vehicles_right}',
        'traffic_intensity_left': traffic_intensity_left,
        'traffic_intensity_right': traffic_intensity_right
    })


if __name__ == "__main__":
    app.run(debug=True)
