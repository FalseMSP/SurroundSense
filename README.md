SurroundSense is CPSE Project to assist Blind or Low Vision people in performing day-to-day tasks independently.

Current Features include:
- Object Detection
- Distance Detection
- TTS
- Reading Signs

The main file is readDualCam.py
Just run that at startup with a speaker and two cameras plugged in.

If you haven't set up the configuration file, run stereothingy.py


**Full instalation Guide:**
Step 1:
Acquire:
2 Logitech c270s or high resolution USB cameras
Nvidia Jetson Nano or other microcomputer
A speaker with some way of connecting to aforementioned microcomputer
A piece of paper with a checkerboard pattern
Step 2:
Upload Ubuntu preferably onto microcomputer
You can use Balena Etcher

Step 3:
Install python3.11
pip install pytorch, numpy, opencv-python, ultralytics, pyttsx3, simpleaudio, pytesseract
also use your package installer to get tesseract-ocr
```sudo apt get tessearct```


Step 4:
Plug in cameras and speaker
Run stereothingy.py
Follow instructions shown in the program
Step 5:
Open readDualCam.py and change “calibration_path” to the path the calibration file created by stereothingy.py
Also change FOCAL_LENGTH and DIST_BETWEEN_CAMERAS to match your hardware specs
Step 6:
Set readDualCam.py to be run on startup
You did it!!
Woohoo
I'm so proud of you!
