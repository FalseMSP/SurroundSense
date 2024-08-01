import numpy as np
import simpleaudio as sa
from ultralytics import YOLO
import cv2
import math
import time
from gtts import gTTS
from playsound import playsound
import os

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

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
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

# Time tracking
last_print_time = time.time()

def get_object_message(object_counts):
    # Sort objects by count and importance (closer to center and larger size)
    sorted_objects = sorted(object_counts.items(), key=lambda x: (-x[1]['count'], -x[1]['size']))

    # Build message
    messages = []
    for obj_name, details in sorted_objects[:3]:  # Limit to top 3 objects
        if details['count'] > 2:
            message = f"Many {obj_name} ahead"
        else:
            directions = ", ".join(details['directions'])
            message = f"{obj_name} {directions}"
        messages.append(message)

    return " and ".join(messages)

while True:
    for i in range(0,10):
        cap.read()
    success, img = cap.read()
    if not success:
        break
    results = model(img, stream=True)

    # Center of the image
    img_center_x = img.shape[1] // 2
    img_center_y = img.shape[0] // 2

    # Object tracking
    object_counts = {}

    highest_object_y = img.shape[0]  # Initialize with max value (bottom of the image)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            obj_name = classNames[int(box.cls[0])]  # Move obj_name definition here

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Center of the bounding box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # Calculate distance to the center of the image
            distance = math.sqrt((box_center_x - img_center_x) ** 2 + (box_center_y - img_center_y) ** 2)

            # Determine direction
            directions = []
            if box_center_x < img_center_x - 100:
                directions.append("to your left")
            elif box_center_x > img_center_x + 100:
                directions.append("to your right")
            if box_center_y < img_center_y - 100:
                directions.append("above you")
            elif box_center_y > img_center_y + 100 and obj_name not in ["diningtable", "sofa", "bed"]:
                directions.append("below you")
            if not directions:
                directions.append("in front of you")

            # Initialize object details if not already present
            if obj_name not in object_counts:
                object_counts[obj_name] = {'count': 0, 'directions': [], 'size': width * height}

            object_counts[obj_name]['count'] += 1
            # Update directions only if it's a new entry or if the direction is new
            object_counts[obj_name]['directions'] = list(set(object_counts[obj_name]['directions'] + directions))
            object_counts[obj_name]['size'] = max(object_counts[obj_name]['size'], width * height)

            # Track the highest object detected
            if box_center_y < highest_object_y:
                highest_object_y = box_center_y


    # Generate speech message
    current_time = time.time()
    if current_time - last_print_time >= 3:
        message = get_object_message(object_counts)
        if message:
            print(message)
            speak(message)
            last_print_time = current_time

    # Play tone based on the highest detected object
    play_tone(highest_object_y, img.shape[0], depth=0.5)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exiting normally.")
