import cv2
import time

# Open the default camera (usually the laptop's built-in camera)
cap = cv2.VideoCapture(0)

# Set the frame width and height if desired
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Counter for unique filenames
image_counter = 0

print("Capturing images every second. Press 'q' to stop.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is captured successfully
        if ret:
            
            cv2.imshow("Camera", frame)
            # Create a filename with the current counter
            filename = f"captured_image_{image_counter}.jpg"

            # Save the current frame as an image file
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")

            # Increase the counter for the next image
            image_counter += 1

            # Wait for 1 second
            time.sleep(1)
        else:
            print("Error: Could not read frame.")
            break

        # Check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

finally:
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
