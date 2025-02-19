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
    # Calculate distance using similar triangles formula
    distance = (H_real * focal_length) / H_image
    return distance

# Example usage

# Step 1: Define the real-world height of the object (in meters)
H_real = 0.5  # Example: object height in meters

# Step 2: Define your camera's focal length (in pixels)
focal_length = 1000  # Example: camera focal length in pixels

# Step 3: Detect the object in your image and get the height in pixels
# Assuming you have already detected the object and got the bounding box (x, y, w, h)
# You can get `h` from your detection (the height of the bounding box)
H_image = 200  # Example: object height in pixels from your detection

# Step 4: Calculate the distance
distance = calculate_distance(H_real, H_image, focal_length)

# Step 5: Output the result
print(f"Distance from the camera to the object: {distance:.2f} meters")





# import cv2
# import numpy as np
#
#
# def detect_object_and_measure_distance(image_path, H_real, focal_length):
#     """
#     Detect the object in the image and calculate the distance from the camera to the object.
#
#     Args:
#     - image_path (str): Path to the input image.
#     - H_real (float): Real-world height of the object (in meters).
#     - focal_length (float): Camera's focal length (in pixels).
#
#     Returns:
#     - distance (float): Distance from the camera to the object (in meters).
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply thresholding to find the object (simple threshold, can be replaced by better object detection)
#     _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Get the bounding box of the largest contour (assuming the object is the largest one)
#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)
#
#     # Draw the bounding box on the image
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Measure the size of the object in the image (height of bounding box)
#     H_image = h  # Height of the bounding box in pixels
#
#     # Calculate the distance to the object
#     distance = calculate_distance(H_real, H_image, focal_length)
#
#     # Display the result
#     cv2.putText(image, f"Distance: {distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # Show the image
#     cv2.imshow("Detected Object", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return distance
#
#
# # Example usage
# image_path = 'otargets1.jpeg'  # Path to your image
# H_real = 0.5  # Real height of the object in meters
# focal_length = 1000  # Camera's focal length in pixels
#
# # Detect object and calculate the distance
# distance = detect_object_and_measure_distance(image_path, H_real, focal_length)
# print(f"Distance from the camera to the object: {distance:.2f} meters")
