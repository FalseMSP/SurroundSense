from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
from gtts import gTTS
from playsound import playsound
import os

DIST_BETWEEN_CAMERAS = 0.170  # In Meters
calibration_path = "/home/bob/SurroundSense/calibration/01-08-24-10-16-rms-1.43-zed-0-ximea-0/"

# Calibration and Rectification parameters (example values, replace with actual calibration data)
os.chdir(calibration_path)
mapL1 = np.load('mapL1.npy')
mapL2 = np.load('mapL2.npy')
mapR1 = np.load('mapR1.npy')
mapR2 = np.load('mapR2.npy')

# Initialize two cameras
cap1 = cv2.VideoCapture(1)
cap1.set(3, 640)
cap1.set(4, 480)

cap2 = cv2.VideoCapture(0)
cap2.set(3, 640)
cap2.set(4, 480)

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
    # model.to('cuda')  # Uncomment if using CUDA
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Function to speak text using gTTS
def speak(text):
    try:
        print(f"Starting to speak: {text}")
        tts = gTTS(text=text, lang='en', slow=False)
        filename = 'temp_audio.mp3'
        tts.save(filename)
        playsound(filename)
        os.remove(filename)  # Clean up the temporary audio file
        print("Finished speaking")
    except Exception as e:
        print(f"Exception during speech: {e}")


# set up defaults for disparity calculation

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

keep_processing = True
apply_colourmap = False

# Time tracking
last_print_time = time.time()

def process_frame(img, results, disparity):
    global last_print_time

    img_center_x = img.shape[1] // 2
    img_center_y = img.shape[0] // 2
    min_distance = float('inf')
    closest_box = None
    closest_class = None
    closest_confidence = None

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            distance = math.sqrt((box_center_x - img_center_x) ** 2 + (box_center_y - img_center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_box = (x1, y1, x2, y2)
                closest_confidence = math.ceil((box.conf[0] * 100)) / 100
                closest_class = int(box.cls[0])

    if closest_box is not None:
        x1, y1, x2, y2 = closest_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(img, f"{classNames[closest_class]} {closest_confidence}", org, font, fontScale, color, thickness)

        # Calculate disparity for closest box
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        disparity_value = disparity[box_center_y, box_center_x]
        focal_length = 0.004  # in meters
        baseline = DIST_BETWEEN_CAMERAS  # Distance between cameras in meters
        if disparity_value > 0:
            distance = (focal_length * baseline) / (disparity_value)  # Prevent division by zero
            object_name = classNames[closest_class]
            print(f"Distance to {object_name}: {distance:.2f} meters")

            # Print and say the object name every 3 seconds
            current_time = time.time()
            if current_time - last_print_time >= 1:
                print(f"{object_name} ahead")
                speak(f"{object_name} ahead")
                last_print_time = current_time

while True:
    for i in range(0, 25):
        cap1.read()
        cap2.read()
    success1, frameL = cap1.read()
    success2, frameR = cap2.read()

    if not success1 or not success2:
        print("Error: Could not read from one of the cameras.")
        break

    # Undistort and rectify images
    undistorted_rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    undistorted_rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    # Convert images to grayscale for stereo matcher
    gray1 = cv2.cvtColor(undistorted_rectifiedL, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(undistorted_rectifiedR, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereoProcessor.compute(gray1, gray2)  # Ensure disparity is float32 .astype(np.float32)
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)

    # Normalize disparity for visualization
    disparity_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)

    # Compute depth map from disparity map
    focal_length = 600  # Assuming same for both cameras
    baseline = DIST_BETWEEN_CAMERAS  # Distance between cameras in meters
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    depth_map[disparity > 0] = (focal_length * baseline) / (disparity[disparity > 0] + 1e-5)  # Prevent division by zero

    # Normalize depth map for visualization
    depth_map_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_display = np.uint8(depth_map_display)  # Convert to 8-bit image
    depth_map_display = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)  # Apply a color map

    # Process both frames with YOLO
    results1 = model(undistorted_rectifiedL, stream=True)
    results2 = model(undistorted_rectifiedR, stream=True)

    # Process both frames
    process_frame(undistorted_rectifiedL, results1, disparity)
    process_frame(undistorted_rectifiedR, results2, disparity)

    # Display the results
    cv2.imshow('Camera 1', undistorted_rectifiedL)
    cv2.imshow('Camera 2', undistorted_rectifiedR)
    cv2.imshow('Disparity Map', disparity_display)  # Show disparity map
    cv2.imshow('Depth Map', depth_map_display)  # Show colorized depth map

    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Exiting normally.")

