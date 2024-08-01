import cv2
import numpy as np
import glob

# Calibration parameters
CHECKERBOARD = (9, 6)  # Number of inner corners per checkerboard row and column
CAMERA1_CALIBRATION_IMAGES = 'camera1R/*.jpg'  # Path to images from camera 1
CAMERA2_CALIBRATION_IMAGES = 'camera2L/*.jpg'  # Path to images from camera 2

# Termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (CHECKERBOARD[0]-1,CHECKERBOARD[1]-1,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real world space
imgpoints1 = []  # 2D points in image plane for camera 1
imgpoints2 = []  # 2D points in image plane for camera 2

def find_corners(images_path, imgpoints_list):
    images = glob.glob(images_path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_list.append(corners2)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f"Checkerboard corners not found in {fname}")

    cv2.destroyAllWindows()

print("Finding corners for camera 1...")
find_corners(CAMERA1_CALIBRATION_IMAGES, imgpoints1)

print("Finding corners for camera 2...")
find_corners(CAMERA2_CALIBRATION_IMAGES, imgpoints2)

# Calibrate the cameras
print("Calibrating camera 1...")
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray.shape[::-1], None, None)
print("Calibrating camera 2...")
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2,
    gray.shape[::-1], criteria=criteria, flags=flags
)

print(f"Stereo calibration successful: {ret}")

# Save calibration results
np.savez('stereo_calibration_data.npz', 
         mtx1=mtx1, dist1=dist1, 
         mtx2=mtx2, dist2=dist2, 
         R=R, T=T, E=E, F=F)

print("Calibration data saved.")

# You can also compute rectification maps if needed
# These maps are used to rectify images to make them easier to compare
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    mtx1, dist1, mtx2, dist2, gray.shape[::-1], R, T, alpha=0
)
map1x1, map1y1 = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, gray.shape[::-1], cv2.CV_16SC2)
map2x2, map2y2 = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, gray.shape[::-1], cv2.CV_16SC2)

print("Rectification maps computed.")
