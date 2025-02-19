import cv2
import numpy as np


def calculate_distance(H_real, H_image, focal_length):
    """
    Calculate the distance from the camera to the object based on the real-world height,
    the height in the image, and the camera's focal length.

    Args:
    - H_real (float): Real-world height of the object (in meters).
    - H_image (float): Height of the object in the image (in pixels).
    - focal_length (float): Camera's focal length (in pixels).

    Returns:
    - distance (float): Distance from the camera to the object (in meters).
    """
    if H_image == 0:  # Prevent division by zero
        return float('inf')

    distance = (H_real * focal_length) / H_image
    return distance


# Open webcam
webcam = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Step 1: Define the real-world height of the object (in meters)
H_real = 3  # Example: object height in meters

# Step 2: Define the camera's focal length (in pixels)
focal_length = 1000  # Example value

# Start video processing loop
while True:
    # Capture frame from the webcam
    ret, imageFrame = webcam.read()

    if not ret or imageFrame is None:  # Check if the frame was captured successfully
        print("Error: Failed to capture image.")
        continue  # Skip this iteration and try again

    # Convert frame from BGR to HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Define color ranges for detection (lower bound, upper bound, and RGB color for display)
    color_ranges = {
        "Red": ([136, 87, 111], [180, 255, 255], (0, 0, 255)),  # Red
        "Yellow": ([20, 100, 100], [30, 255, 255], (0, 255, 0)),  # Yellow
        "Blue": ([94, 80, 2], [120, 255, 255], (255, 0, 0))  # Blue
    }

    # Define kernel for dilation to enhance mask detection
    kernel = np.ones((5, 5), "uint8")

    # Loop through each color for detection
    for color, (lower, upper, rgb) in color_ranges.items():
        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)

        # Create a mask for the specific color
        mask = cv2.inRange(hsvFrame, lower, upper)
        mask = cv2.dilate(mask, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter out small areas (noise)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), rgb, 2)

                # Calculate distance using detected object height in pixels
                H_image = h
                distance = calculate_distance(H_real, H_image, focal_length)

                # Display the calculated distance on the frame
                cv2.putText(imageFrame, f"{color} Distance: {distance:.2f} m", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, rgb, 2)

                # Print the object details
                print(f"{color} Object area (in pixels): {w * h}")
                print(f"{color} Calculated distance: {distance:.2f} meters")

    # Display the processed frame
    cv2.imshow("Color Detection", imageFrame)

    # Exit program when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
