import cv2
import numpy as np

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not webcam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Start a while loop
while True:
    # Reading the video from the webcam in image frames
    ret, imageFrame = webcam.read()

    # Check if frame was successfully captured
    if not ret or imageFrame is None:
        print("Error: Couldn't read frame.")
        continue  # Skip this iteration and try again

    # Convert the imageFrame in BGR to HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for yellow color and define mask
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Set range for blue color and define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Morphological Transform, Dilation for each color
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # Detect and label red color
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    # Detect and label yellow color
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    # Detect and label blue color
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

    # Display the processed frame
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()