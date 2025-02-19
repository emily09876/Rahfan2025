import cv2
import numpy as np
#test
# Load the image

image = cv2.imread('targets1.jpeg')

if image is None:
    print("Error: Image not loaded. Check the file path.")
    exit()
# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for yellow color in HSV space
lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow

# Create a mask that captures only yellow regions
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Perform bitwise AND to extract the yellow areas from the image
yellow_objects = cv2.bitwise_and(image, image, mask=mask)

# Find contours of the yellow objects
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours around the detected yellow objects
for contour in contours:
    if cv2.contourArea(contour) > 1000:  # Minimum area to filter out small noise
        # Draw the contour in green with thickness 2
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Show the original image with the detected yellow objects
cv2.imshow('Detected Yellow Objects', image)

# Optionally, show the mask and the extracted yellow objects
cv2.imshow('Mask', mask)
cv2.imshow('Yellow Objects', yellow_objects)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()