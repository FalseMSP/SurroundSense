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

while True:
    success, img = cap.read()
    if not success:
        break
    results = model(img, stream=True)

    # Center of the image
    img_center_x = img.shape[1] // 2
    img_center_y = img.shape[0] // 2

    # Find the closest box to the center
    min_distance = float('inf')
    closest_box = None
    closest_class = None
    closest_confidence = None

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Center of the bounding box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            # Calculate distance to the center of the image
            distance = math.sqrt((box_center_x - img_center_x) ** 2 + (box_center_y - img_center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_box = (x1, y1, x2, y2)
                closest_confidence = math.ceil((box.conf[0] * 100)) / 100
                closest_class = int(box.cls[0])

    if closest_box is not None:
        x1, y1, x2, y2 = closest_box

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(img, f"{classNames[closest_class]} {closest_confidence}", org, font, fontScale, color, thickness)

        # Print and say the object name every 3 seconds
        current_time = time.time()
        if current_time - last_print_time >= 3:
            object_name = classNames[closest_class]
            print(f"{object_name} ahead")
            speak(f"{object_name} ahead")
            last_print_time = current_time

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exiting normally.")
