from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import pyttsx3
from playsound import playsound
import os
import threading
import simpleaudio as sa
import datetime
#import smbus

# I2C bus
#bus = smbus.SMBus(1)  # For Jetson boards, /dev/i2c-1 is typically used

# PCF8591 address
#PCF8591_ADDRESS = 0x48

# Constants
FOCAL_LENGTH = 930.3284425137067  # In PIXELS
DIST_BETWEEN_CAMERAS = 0.11945  # In meters
the_T = np.array([DIST_BETWEEN_CAMERAS, 0, 0], dtype=np.float64)
calibration_path = "/home/bob/SurroundSense/calibration/08-08-24-15-05-rms-1.15-zed-0-ximea-0"

# init flags
swapLeft = False
disableProg = False
TTS = True
beeper = True
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

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

lastSpeak = ""

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
ignore_list = [
    "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "umbrella",
    "suitcase", "frisbee", "skis", "snowboard", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "toilet",
    "microwave", "oven", "toaster", "sink", "refrigerator", "clock", "vase",
    "teddy bear", "hair drier", "toothbrush"
]

# Function to speak text using gTTS
def speak(text):
    try:
        print(f"Starting to speak: {text}")

        # Set properties if needed
        engine.setProperty('rate', 150)    # Speed percent (can go over 100)
        engine.setProperty('volume', 1)     # Volume 0-1

        # Define a function to speak text
        def speak_text(text):
            engine.say(text)
            engine.runAndWait()
            print("Finished speaking")
        
        # Create a thread to speak the text
        threading.Thread(target=speak_text, args=(text,), daemon=True).start()
        
    except Exception as e:
        print(f"Exception during speech: {e}")

#Stair Detection
def detect_stairs(frame):
    # Get image dimensions
    height, width = frame.shape[:2]

    # Define the region of interest (bottom 10% of the image)
    bottom_region = int(height * 0.40)
    roi = frame[-bottom_region:, :]  # Crop the bottom 10%

    # Convert roi to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # Adjust the threshold if needed

    # Draw only horizontal lines on the original frame
    result_frame = frame.copy()

    if lines is not None:
        # Store lines that are considered horizontal
        horizontal_lines = []
        
        for rho, theta in lines[:, 0]:
            # Convert polar coordinates (rho, theta) to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # Compute the angle of the line in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize angle to be between 0 and 180
            angle = np.abs(angle % 180)
            
            # Check if the line is approximately horizontal
            if (angle < 10 or angle > 170):
                # Translate line coordinates to match the original frame
                y1 += height - bottom_region
                y2 += height - bottom_region
                horizontal_lines.append((x1, y1, x2, y2))
                # Draw the line on the result frame
                cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Determine if there are at least 5 parallel lines that are not too close to each other
        if len(horizontal_lines) >= 5:
            # Sort lines by their vertical intercept (y0 + b * x0) to approximate their positions
            def get_y_intercept(line):
                x1, y1, x2, y2 = line
                return (y1 + y2) / 2
            
            horizontal_lines.sort(key=get_y_intercept)

            # Check the distances between consecutive lines
            min_lines_count = 5
            min_distance = 30  # Minimum distance between lines to consider them separate
            count_valid_lines = 1  # At least one line is valid initially
            last_y_intercept = get_y_intercept(horizontal_lines[0])
            
            for i in range(1, len(horizontal_lines)):
                current_y_intercept = get_y_intercept(horizontal_lines[i])
                if abs(current_y_intercept - last_y_intercept) >= min_distance:
                    count_valid_lines += 1
                    last_y_intercept = current_y_intercept
            
            if count_valid_lines >= min_lines_count:
                speak("Stairs Ahead")
    
    return result_frame, edges
# Function to play a tone based on object height and depth
def play_tone(y_position, image_height, duration=250, depth=0.5):
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
    volume = depth * 0.4
    audio = (audio * volume).astype(np.int16)

    # Play the tone
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()
# Set up defaults for disparity calculation
max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 4,  # Must be divisible by 16
    blockSize=4,
    P1=8 * 3 * 4 ** 2,  # Default value: 8*3*blockSize^2
    P2=32 * 3 * 4 ** 2,  # Default value: 32*3*blockSize^2
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

keep_processing = True
apply_colourmap = False

# Time tracking
last_print_time = time.time()

# Joystick functions
def read_analog_input(channel):
    if channel < 0 or channel > 3:
        raise ValueError("Channel must be between 0 and 3")
    # Command to select the channel (0x40 is the base command, add channel number)
    bus.write_byte(PCF8591_ADDRESS, 0x40 | channel)
    time.sleep(0.1)  # Wait for the conversion to complete
    return bus.read_byte(PCF8591_ADDRESS)  # Read the ADC value

def direction():
    # Read analog values from channels
    x_value = read_analog_input(0)
    y_value = read_analog_input(2)
    button_value = read_analog_input(1)
    
    # Determine if the button is pressed
    button_pressed = button_value <= 30
    
    # Determine the joystick direction
    if x_value <= 30:
        x = -1  # Left
    elif x_value >= 225:
        x = 1   # Right
    else:
        x = 0

    if y_value <= 30:
        y = -1  # Up
    elif y_value >= 225:
        y = 1   # Down
    else:
        y = 0
    
    # Return (x, y) coordinates and button pressed state
    return (x, y, button_pressed)


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
    global lastSpeak

    img_center_x = img.shape[1] // 2
    img_center_y = img.shape[0] // 2
    min_distance = float('inf')
    closest_box = None
    closest_class = None
    closest_confidence = None
    detected_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            distance = math.sqrt((box_center_x - img_center_x) ** 2 + (box_center_y - img_center_y) ** 2)
            if classNames[int(box.cls[0])] == "clock":
                currtime = datetime.datetime.now()
                currtime = currtime.strftime("%H:%M")
                if TTS:
                    speak(f"The time is currently {currtime}")

            if distance < min_distance and (classNames[int(box.cls[0])] not in ignore_list):
                min_distance = distance
                closest_box = (x1, y1, x2, y2)
                closest_confidence = math.ceil((box.conf[0] * 100)) / 100
                closest_class = int(box.cls[0])
                detected_objects.append(closest_class)

    # Process and announce the most important object not in the ignore list
    if closest_box is not None:
        x1, y1, x2, y2 = closest_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        object_name = classNames[closest_class]

        if object_name not in ignore_list:
            cv2.putText(img, f"{object_name} {closest_confidence}", org, font, fontScale, color, thickness)

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
                    realdist = distance
                    distance /= 5 # Max Loudness is 1
                    if lastSpeak == f"{object_name} {realdist:.0f} meters ahead":
                        TTS = False
                    else:
                        TTS = True
                    lastSpeak = f"{object_name} {realdist:.0f} meters ahead"
                    if distance >= 1:
                        distance = 1 # Clamping for the volume of tone
                    distance = 1-distance
                    if beeper:
                        play_tone(central_y1,img.shape[0],250,distance)
                    if realdist < 1:
                        print(f"{object_name} {realdist:.2f} meters ahead")
                        if TTS:
                            speak(f"{object_name} less than a meter ahead")
                        return
                    if realdist > 7:
                        return
                    # Print and say the object
                    print(f"{object_name} {realdist:.2f} meters ahead")
                    if TTS:
                        speak(f"{object_name} {realdist:.0f} meters ahead")
def disparity_to_distance(disparity, focal_length, baseline):
    """
    Convert disparity value to distance.

    :param disparity: Disparity value.
    :param focal_length: Focal length of the camera.
    :param baseline: Distance between the cameras.
    :return: Distance in meters.
    """
    if disparity > 0:
        distance = (focal_length * baseline) / disparity
        return distance
    return float('inf')  # Return a large number if disparity is zero or invalid


def find_close_objects(img, disparity, focal_length, baseline, distance_threshold):
    """
    Find objects that are less than `distance_threshold` meters away in the center of the image.

    :param img: Input image.
    :param disparity: Disparity map.
    :param focal_length: Focal length of the camera.
    :param baseline: Distance between the cameras.
    :param distance_threshold: Distance threshold in meters.
    :return: List of tuples (distance, y_coordinate).
    """
    img_center_x = img.shape[1] // 2
    img_center_y = img.shape[0] // 2
    central_region_width = img.shape[1] * 0.2  # 20% of image width
    central_region_height = img.shape[0] * 0.8  # 80% of image height

    # Define the region of interest (ROI) for the central part of the image
    roi_x1 = int(img_center_x - central_region_width // 2)
    roi_x2 = int(img_center_x + central_region_width // 2)
    roi_y1 = int(img_center_y - central_region_height // 2)
    roi_y2 = int(img_center_y + central_region_height // 2)

    close_objects = []

    # Iterate over the central region to find close objects
    for y in range(max(roi_y1, 0), min(roi_y2, img.shape[0])):
        for x in range(max(roi_x1, 0), min(roi_x2, img.shape[1])):
            disparity_value = disparity[y, x]
            distance = disparity_to_distance(disparity_value, focal_length, baseline)
            if distance < distance_threshold:
                close_objects.append((distance, y, x))

    return close_objects

# Initialize a Lock object
thread_lock = threading.Lock()

def proccessResults(undistorted_rectifiedL, disparity):
    # Acquire the lock
    with thread_lock:
        # Process both frames with YOLO
        results1 = model(undistorted_rectifiedL, stream=True)
        # results2 = model(undistorted_rectifiedR, stream=True)

        # Speak + Print Results
        detect_stairs(undistorted_rectifiedL)
        process_frame(undistorted_rectifiedL, results1, disparity)
        # process_frame(undistorted_rectifiedR, results2, disparity)

button_pressed = False
x = 0
y = 0

while True:
    # x, y, button_pressed = direction()
    # Handle on/off through joystick press
    if button_pressed == True:
        disableProg = not disableProg
        # Release Camera
        if disableProg:
            cap1.release()
            cap2.release()
            speak("Power Off")
        else: # Reactivate Camera
            cap1 = cv2.VideoCapture(1)
            cap1.set(3, 640)
            cap1.set(4, 480)
            
            cap2 = cv2.VideoCapture(0)
            cap2.set(3, 640)
            cap2.set(4, 480)
            speak("Power On")
            TTS = True
            beeper = True
        time.sleep(2)
    if disableProg:
        time.sleep(0.1)
        continue;

    if x == 1:
        # Right, TTS ON
        TTS = True
        speak("Speaking on")
    elif x == -1:
        # LEFT, TTS OFF
        TTS = False
        speak("Speaking off")

    if y == 1:
        # UP, Beeper OFF
        beeper = True
        speak("Beeper On")
    elif y == -1:
        # DOWN, Beeper ON
        beeper = False
        speak("Beeper Off")
    # Remove buffer from camera feeed
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
    #disparity = cv2.medianBlur(disparity, 5)
    
    # Apply speckle filter
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)

    # Normalize disparity for visualization
    disparity_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)

    # Start a new thread to process results
    with thread_lock:  # Ensure no other thread is running
        threading.Thread(target=proccessResults, args=(undistorted_rectifiedL, disparity,), daemon=True).start()

    # Display the Feeds
    cv2.imshow('Camera 1', undistorted_rectifiedL)
    cv2.imshow('Camera 2', undistorted_rectifiedR)
    cv2.imshow('Disparity Map', disparity_display)  # Show disparity map
    disparity_colour_mapped = cv2.applyColorMap(
            (disparity_display * (256. / max_disparity)).astype(np.uint8),
            cv2.COLORMAP_HOT)
    cv2.imshow('Color', disparity_colour_mapped)
    
    if cv2.waitKey(1) == ord('s'):
        swapLeft = not swapLeft
    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Exiting normally.")

