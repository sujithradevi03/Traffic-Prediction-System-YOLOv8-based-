import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# Define the threshold for considering traffic as heavy
heavy_traffic_threshold = 10

# Define the vertices for the quadrilaterals
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Define the vertical range for the slice and lane threshold
x1, x2 = 325, 635 
lane_threshold = 609

# Define the positions for text annotations on the image
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

# Define font, scale, and colors for the annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White color for text
background_color = (0, 0, 255)  # Red background for text

# Function to detect weather based on different visual conditions
def detect_weather(frame, prev_gray):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Brightness (for sun/cloud detection)
    brightness = np.mean(gray)

    # 2. Edge Detection for Fog
    edges = cv2.Canny(frame, 100, 200)
    edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1])

    # 3. Rain or Snow Detection (Motion Analysis)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_intensity = np.mean(magnitude)

    # 4. Reflection Analysis for Wet Road (detect rain reflection)
    reflection = cv2.inRange(hsv, (0, 0, 230), (180, 255, 255))  # Detect bright reflections
    reflection_density = np.sum(reflection) / (frame.shape[0] * frame.shape[1])

    # 5. Sky Detection for Clear/Cloudy Sky
    sky_color_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    sky_area = np.sum(sky_color_mask) / (frame.shape[0] * frame.shape[1])

    # Mapping weather conditions to their respective values
    weather_conditions = {
        "Rainy": motion_intensity * reflection_density,    # Higher motion + reflections = Rain
        "Foggy": 1 - edge_density,                         # Lower edge density = Fog
        "Cloudy": 1 - sky_area,                            # Less sky detected = Cloudy
        "Clear Sky": sky_area,                             # More sky detected = Clear Sky
        "Low Visibility": 1 - edge_density,                # Low edge visibility = Poor Visibility
        "Sunny": brightness / 255                          # Higher brightness = Sunny
    }

    # Output the weather condition with the highest value
    predicted_weather = max(weather_conditions, key=weather_conditions.get)
    return predicted_weather, gray  # Return current gray frame for next iteration

# Function to display the weather prediction in the top center of the frame
def display_weather_in_top_center(frame, text, font, font_scale, font_color):
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    frame_width = frame.shape[1]
    text_x = (frame_width - text_size[0]) // 2  # Center horizontally
    text_y = 50  # Set a fixed position for top center

    # Display weather text with a background rectangle
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                  (text_x + text_size[0] + 10, text_y + 10), background_color, -1)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, 2, cv2.LINE_AA)

# Open the video
cap = cv2.VideoCapture('sample_video.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_sample_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Cannot read the first frame of the video.")
    cap.release()
    sys.exit(1)

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot read the frame.")
        break

    # Create a copy of the original frame to modify
    detection_frame = frame.copy()

    # Black out the regions outside the specified vertical range
    detection_frame[:x1, :] = 0  # Black out from top to x1
    detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame

    # Perform inference on the modified frame
    results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
    processed_frame = results[0].plot(line_width=1)

    # Restore the original top and bottom parts of the frame
    processed_frame[:x1, :] = frame[:x1, :].copy()
    processed_frame[x2:, :] = frame[x2:, :].copy()

    # Draw the quadrilaterals on the processed frame
    cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

    # Retrieve the bounding boxes from the results
    bounding_boxes = results[0].boxes

    # Initialize counters for vehicles in each lane
    vehicles_in_left_lane = 0
    vehicles_in_right_lane = 0

    # Loop through each bounding box to count vehicles in each lane
    for box in bounding_boxes.xyxy:
        if box[0] < lane_threshold:
            vehicles_in_left_lane += 1
        else:
            vehicles_in_right_lane += 1

    # Determine the traffic intensity for each lane
    traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
    traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

    # Detect the weather conditions
    predicted_weather, prev_gray = detect_weather(frame, prev_gray)

    # Display the weather prediction text at the top center of the frame
    display_weather_in_top_center(processed_frame, predicted_weather, font, font_scale, font_color)

    # Display traffic intensity for each lane
    cv2.putText(processed_frame, f'Left Lane: {traffic_intensity_left}', 
                (10, 80), font, font_scale, font_color, 2, cv2.LINE_AA)
    cv2.putText(processed_frame, f'Right Lane: {traffic_intensity_right}', 
                (820, 80), font, font_scale, font_color, 2, cv2.LINE_AA)

    # Display number of vehicles in each lane below traffic intensity
    cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', 
                (10, 120), font, font_scale, font_color, 2, cv2.LINE_AA)
    cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', 
                (820, 120), font, font_scale, font_color, 2, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(processed_frame)

    # Show the frame (optional)
    cv2.imshow('Traffic Analysis', processed_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
