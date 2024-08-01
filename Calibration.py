import cv2
import numpy as np
import glob
import os

# Calibration parameters
CHECKERBOARD = (8, 6)  # Number of internal corners in the checkerboard
SQUARE_SIZE = 0.02315  # Size of a square in meters (adjust as needed)

# Prepare object points based on the checkerboard pattern size
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Arrays to store object points and image points from all the images
objpoints = []
imgpoints = []

# Read images from the directory
images = glob.glob('calibration_images/*.jpg')  # Change path if needed

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration data
calibration_data = {
    'cameraMatrix': cameraMatrix.tolist(),
    'distCoeffs': distCoeffs.tolist(),
    'rvecs': [rvec.tolist() for rvec in rvecs],
    'tvecs': [tvec.tolist() for tvec in tvecs]
}

with open('camera_calibration_data.json', 'w') as f:
    import json
    json.dump(calibration_data, f, indent=4)

print("Calibration complete. Calibration data saved to 'camera_calibration_data.json'.")

