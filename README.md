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
_Step 1_:
Acquire:
2 Logitech c270s or high resolution USB cameras
Nvidia Jetson Nano or other microcomputer
A speaker with some way of connecting to aforementioned microcomputer
A piece of paper with a checkerboard pattern

_Step 2_:
Upload Ubuntu preferably onto microcomputer
You can use Balena Etcher

_Step 3_:
Install python3.11
pip install pytorch, numpy, opencv-python, ultralytics, pyttsx3, simpleaudio, pytesseract
also use your package installer to get tesseract-ocr: 
```sudo apt get tessearct```


_Step 4_:
Plug in cameras and speaker
Run stereothingy.py
Follow instructions shown in the program
_Step 5_:
Open readDualCam.py and change “calibration_path” to the path the calibration file created by stereothingy.py
Also change FOCAL_LENGTH and DIST_BETWEEN_CAMERAS to match your hardware specs
_Step 6_:
Set readDualCam.py to be run on startup
You did it!!
Woohoo
I'm so proud of you!
