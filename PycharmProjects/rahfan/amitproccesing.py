import cv2
import numpy as np

# Open the video file
video = cv2.VideoCapture('amit.mp4')

# Check if the video file opened successfully
if not video.isOpened():
    print("Error: Unable to open video file")
    exit()

# Define the kernel for morphological transformations
kernel = np.ones((5, 5), "uint8")

# Start processing the video frames
while True:
    # Read a frame
    ret, imageFrame = video.read()

    # Check if the frame was read successfully
    if not ret:
        print("End of video or cannot read the frame")
        break

    # Convert the image frame to HSV
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Define color ranges and create masks
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    yellow_lower = np.array([23, 100, 100], np.uint8)
    yellow_upper = np.array([28, 255, 250], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    blue_lower = np.array([100, 150, 0], np.uint8)
    blue_upper = np.array([140, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Apply morphological transformations
    red_mask = cv2.dilate(red_mask, kernel)
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # Detect and label red color
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Detect and label yellow color
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Yellow Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Detect and label blue color
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Blue Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Color Detection", imageFrame)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()