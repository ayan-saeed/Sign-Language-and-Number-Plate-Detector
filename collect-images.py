# Import the 'os' module for interacting with the operating system (paths, folders, files)
import os
# Import the 'cv2' module, a real-time computer vision library for python, allowing tasks like webcam capture and drawing on frames
import cv2
# # Import time so we can add short delays (e.g., a delay before capturing)
import time

# Defines the root folder where all captured images will be saved
DATA_DIR = 'data'
# Builds a list of class labels A–Z using ASCII codes; chr() converts code → character
CLASS_LABELS = []
for i in range(ord('A'), ord('Z') + 1):
    CLASS_LABELS.append(chr(i))
# Counts how many classes we have (26 for A–Z); used to drive the outer loop
NUM_CLASSES = len(CLASS_LABELS)
# How many images to capture per class/letter; more data = greater model robustness
IMAGES_PER_CLASS = 100
# Specifies Which camera device to open:
# 0 is the default/built-in camera; 1,2... are additional cameras
CAMERA_INDEX = 0

# Creates the root data directory if it doesn't exist; exist_ok avoids errors if it already exists
os.makedirs(DATA_DIR, exist_ok=True)
# Creates a VideoCapture object that connects to the webcam so we can read frames
cap = cv2.VideoCapture(CAMERA_INDEX)

# Verifies the camera opened successfully; if not, inform the user and exit to avoid crashes later
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
# Prints a one-time instruction that the user can stop at any moment by pressing 'S', in the command line/terminal
print('Press "S" at any time to stop the camera and exit.\n')

# Iterates over every class index (0..NUM_CLASSES-1); this controls which letter we’re collecting
for class_id in range(NUM_CLASSES):
    class_label = CLASS_LABELS[class_id]
    # Builds the folder path for this specific letter (e.g., data/A, data/B, ...)
    class_path = os.path.join(DATA_DIR, class_label)
    # Ensures this letter’s folder exists so we can write image files into it
    os.makedirs(class_path, exist_ok=True)
    # Prints progress so the user knows which letter is next and where they are in the sequence
    print(f"\nPreparing to collect data for letter '{class_label}' ({class_id + 1}/{NUM_CLASSES})")
    print('Press "Q" when ready to start capturing...')

    # Shows a live preview until the user presses 'Q' to start or 'S' to stop all.
    while True: 
        # Reads one frame from the camera; ret=True if successful, and frame contains the image
        ret, frame = cap.read()
        # If frame grab failed (e.g., camera unplugged), notify and break out
        if not ret: 
            print("Failed to grab frame.")
            break
        # Flips the frame horizontally so it isn't inverted
        frame = cv2.flip(frame, 1)
        # Draws a line of text with instrcutions near the top
        cv2.putText(frame, 'Press "Q" to start capturing | Press "S" to stop', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        # Draws the current letter so the user knows what they’re about to record
        cv2.putText(frame, f'Letter: {class_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # Names the window
        cv2.imshow("Sign Language Data Capture", frame)
        # cv2.waitKey waits for keyboard input (in ms), 
        # and '& 0xFF' standardizes the keycode across platforms
        key = cv2.waitKey(25) & 0xFF
        # Break out the capture loop, as the user is ready
        if (key == ord('q')): 
            break
        # Stops the entire program immediately, when the user presses 's'
        elif key == ord('s'):
            print("\nCamera stopped by user.")
            # Releases the camera device so other apps can use it and to avoid locks
            cap.release()
            # Closes all OpenCV windows to clean up the GUI
            cv2.destroyAllWindows()
            # Exits the Python process to stop everything
            exit()
    # Adds a small delay
    time.sleep(2)
    # Tells the user how many frames will be captured for this letter
    print(f"Collecting {IMAGES_PER_CLASS} images for letter '{class_label}'...")
    counter = 1
    # Loops until the desired number of images for this letter has been saved
    while counter <= IMAGES_PER_CLASS:
        # Reads the next frame from the camera
        ret, frame = cap.read()
        # Abort this letter’s capture if the camera fails mid-stream (prevents corrupted files)
        if not ret:
            print("Camera frame not available.")
            break
        frame = cv2.flip(frame, 1)
        # Draws a bottom-left progress indicator like "23/100" so you know how many images have been captured
        cv2.putText(frame, f'{counter + 1}/{IMAGES_PER_CLASS}', (10, frame.shape[0] - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)
        # Also display the current letter being captured
        cv2.putText(frame, f'Letter: {class_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Sign Language Data Capture", frame)
        # Writes the current frame to disk as a JPEG; filename is the counter (1.jpg, 2.jpg, ..., 100.jpg)
        cv2.imwrite(os.path.join(class_path, f"{counter}.jpg"), frame)
        # Increments the counter
        counter += 1
        # If 'S' is pressed, stop instantly and clean up resources
        if cv2.waitKey(25) & 0xFF == ord('s'):
            print("\nCamera stopped by user.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("\nData collection complete.")
cap.release()
cv2.destroyAllWindows() 