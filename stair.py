import cv2
import numpy as np


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
                print("stairs ahead")
    
    return result_frame, edges

def main():
    # Open a connection to the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame to detect stairs
        result_frame, edges = detect_stairs(frame)

        # Display the result
        cv2.imshow('Detected Stairs', result_frame)
        cv2.imshow('Edges', edges)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

