import cv2
import os
import time

# Define the path for the 'Captures' folder
current_directory = os.path.dirname(os.path.abspath(__file__))
captures_directory = os.path.join(current_directory, 'Captures')

# Create the 'Captures' folder if it does not exist
if not os.path.exists(captures_directory):
    os.makedirs(captures_directory)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

image_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Display the captured frame
    cv2.imshow('Image Capture', frame)

    # Save the frame as an image in the 'Captures' folder
    image_filename = os.path.join(captures_directory, f'capture_{image_count}.png')
    cv2.imwrite(image_filename, frame)
    print(f"Image saved: {image_filename}")

    image_count += 1

    # Wait for 4 seconds before capturing the next image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(4)  # Pause for 4 seconds

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()