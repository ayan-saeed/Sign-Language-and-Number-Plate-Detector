import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
DATA_DIR = './data'
data = []
labels = []
# Iterates over each subfolder inside DATA_DIR
for dir in os.listdir(DATA_DIR):
     # Iterates over every image file inside the current class folder
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        data_aux = []
        # Reads the image from disk as a NumPy array in BGR color space
        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Runs the MediaPipe hand detector on this RGB image to get landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Loops over all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = []
                y_coords = []
                # There are 21 landmarks per hand; iterates through each landmark index
                for i in range(len(hand_landmarks.landmark)):
                    landmark = hand_landmarks.landmark[i]
                    # Extracts x and y (normalized to image width/height) for the current landmark
                    x_val = landmark.x
                    y_val = landmark.y
                    x_coords.append(x_val)
                    y_coords.append(y_val)
                # Finds min and max across all landmarks to define the hand's bounding box in normalized coordinates
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                for i in range(len(hand_landmarks.landmark)):
                    x_raw = hand_landmarks.landmark[i].x
                    y_raw = hand_landmarks.landmark[i].y
                    # Normalizes inside the hand’s own bounding box:
                    x_normalized = (x_raw - x_min) / (x_max - x_min)
                    y_normalized = (y_raw - y_min) / (y_max - y_min)
                    # Appends normalized x and y to the feature vector
                    data_aux.append(x_normalized)
                    data_aux.append(y_normalized)
            # After processing the detected hand, appends the feature vector to `data`
            data.append(data_aux)
            # Appends the associated class label, i.e. folder name
            labels.append(dir)
f = open('data.pickle', 'wb')
pickle.dump({'data': data,
             'labels': labels}, f)
f.close()