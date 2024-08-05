from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
from gtts import gTTS
from playsound import playsound
import os
import threading
import simpleaudio as sa

# Constants
FOCAL_LENGTH = 930.3284425137067  # In PIXELS
DIST_BETWEEN_CAMERAS = 0.11945  # In METERS, this should be the actual distance between your cameras
the_T = np.array([DIST_BETWEEN_CAMERAS, 0, 0], dtype=np.float64)
calibration_path = "/home/bob/SurroundSense/calibration/01-08-24-14-09-rms-0.59-zed-0-ximea-0"

# init flags
swapLeft = True
# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
    # model.to('cuda')  # Uncomment if using CUDA
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)

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
        
        # Define a function to play sound in a separate thread
        def play_sound(filename):
            playsound(filename)
            os.remove(filename)  # Clean up the temporary audio file
            print("Finished speaking")
        
        # Create a thread to play the sound
        threading.Thread(target=play_sound, args=(filename,), daemon=True).start()
        
    except Exception as e:
        print(f"Exception during speech: {e}")

# Function to play a tone based on object height and depth
def play_tone(y_position, image_height, duration=500, depth=0.5):
    """
    Play a tone based on object height and depth.

    :param y_position: The y-coordinate of the detected object.
    :param image_height: The height of the image frame.
    :param duration: The duration of the tone in milliseconds.
    :param depth: The depth factor affecting the volume of the tone (0.0 to 1.0).
    """
    freq_min = 300  # Minimum frequency (Hz)
    freq_max = 1000  # Maximum frequency (Hz)
    freq_range = freq_max - freq_min

    # Normalize y_position to a value between 0 and 1
    normalized_y = y_position / image_height

    # Calculate frequency based on the normalized y_position
    freq = freq_min + (freq_range * (1 - normalized_y))

    # Generate a sine wave tone
    sample_rate = 44100  # Samples per second
    t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
    tone = np.sin(freq * t * 2 * np.pi)

    # Ensure that highest value is in 16-bit range
    audio = tone * (2**15 - 1) / np.max(np.abs(tone))
    audio = audio.astype(np.int16)

    # Adjust volume based on depth
    volume = depth * 0.1
    audio = (audio * volume).astype(np.int16)

    # Play the tone
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()
# Set up defaults for disparity calculation
max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

keep_processing = True
apply_colourmap = False

# Time tracking
last_print_time = time.time()

# Helper for proccess_frame
def mode(a, axis=0):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

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

        # Calculate central 10% region of the bounding box
        box_width = x2 - x1
        box_height = y2 - y1
        central_region_width = int(box_width * 0.1)
        central_region_height = int(box_height * 0.1)

        # Calculate central coordinates
        central_x1 = x1 + (box_width - central_region_width) // 2
        central_y1 = y1 + (box_height - central_region_height) // 2
        central_x2 = central_x1 + central_region_width
        central_y2 = central_y1 + central_region_height

        # Extract disparity values within the central 10% region
        box_disparities = []
        for i in range(central_y1, central_y2):
            for j in range(central_x1, central_x2):
                disparity_value = disparity[i, j]
                if disparity_value > 0:  # Ignore invalid disparity values
                    box_disparities.append(disparity_value)

        if box_disparities:
            # Calculate the median disparity value
            most_frequent, counts = mode(np.array(box_disparities))
            median_disparity = most_frequent[0]
            # Convert baseline from meters to the same unit used in the calculation
            focal_length = FOCAL_LENGTH  # Pixels
            baseline = DIST_BETWEEN_CAMERAS  # Meters

            # Calculate distance in meters
            if median_disparity > 0:
                distance = (focal_length * baseline) / median_disparity  # distance in meters
                distance *= 7 * 1.93934426
                object_name = classNames[closest_class]
                print(f"Distance to {object_name}: {distance:.2f} meters")
                distance /= 5 # Max Loudness is 1
                if distance >= 1:
                    distance = 1 # Clamping for the volume of tone
                distance = 1-distance
                play_tone(central_y1,img.shape[0],250,distance)

                # Print and say the object
                print(f"{object_name} ahead")
                speak(f"{object_name} ahead")


while True:
    for i in range(0, 25):
        cap1.read()
        cap2.read()
    if swapLeft:
        success1, frameL = cap1.read()
        success2, frameR = cap2.read()
    else:
        success1, frameL = cap2.read()
        success2, frameR = cap1.read()
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
    disparity = stereoProcessor.compute(gray1, gray2)  # Ensure disparity is 16-bit signed single-channel
    disparity = disparity.astype(np.int16)  # Ensure the disparity is in the correct format for filterSpeckles
    
    # Apply speckle filter
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)

    # Normalize disparity for visualization
    disparity_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)

    # Process both frames with YOLO
    results1 = model(undistorted_rectifiedL, stream=True)
    results2 = model(undistorted_rectifiedR, stream=True)

    # Display the Feeds
    cv2.imshow('Camera 1', undistorted_rectifiedL)
    cv2.imshow('Camera 2', undistorted_rectifiedR)
    cv2.imshow('Disparity Map', disparity_display)  # Show disparity map
    
    # Speak + Print Results
    process_frame(undistorted_rectifiedL, results1, disparity)
    #process_frame(undistorted_rectifiedR, results2, disparity)
    
    if cv2.waitKey(1) == ord('s'):
        swapLeft = not swapLeft
    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Exiting normally.")

