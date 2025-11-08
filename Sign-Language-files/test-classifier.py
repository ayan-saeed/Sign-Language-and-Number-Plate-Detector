import cv2
import mediapipe as mp
import pickle
import numpy as np

model_file = pickle.load(open('./model.p', 'rb'))
model  = model_file['model']
# Initializes the webcam using OpenCV, with arguement '0' - means use the default laptop camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True: 
    data_aux = []
    x_ = []
    y_ = []
    # Reads a single frame from the webcam
    ret, frame = cap.read()
    # Flips the frame horizontally to create a mirror image
    frame = cv2.flip(frame, 1)
    # If the frame wasn’t captured correctly (e.g., camera disconnected), stop the loop
    if not ret:
        break
    H, W, _ = frame.shape
    # Converts the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Processes the RGB frame with MediaPipe’s hand detector
    results = hands.process(frame_rgb)
    # Create a semi-transparent black rectangle overlay at the top-left
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (250, 35), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Draw text on top of the rectangle
    cv2.putText(frame, 'Press "Q" to stop', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            # Compute pixel-based bounding box corners by scaling normalized coordinates
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)
            x_min, x_max = min(x_), max(x_)
            y_min, y_max = min(y_), max(y_)
            # Re-normalize all coordinates so the hand fits a standard scale
            for landmark in hand_landmarks.landmark:
                x_norm = (landmark.x - x_min) / (x_max - x_min)
                y_norm = (landmark.y - y_min) / (y_max - y_min)
                data_aux.append(x_norm)
                data_aux.append(y_norm)
            # Process only the first detected hand
            break
        prediction = model.predict([np.asarray(data_aux)])
        # Get class probabilities - how confident the model is in each possible letter
        prediction_proba = model.predict_proba([np.asarray(data_aux)])[0]
        # Extract the highest probability
        confidence = np.max(prediction_proba) * 100
        prediction_character = prediction[0]
        # Print the prediction and confidence to the terminal as well on the camera
        print(f"{prediction_character} ({confidence:.2f}%)")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{prediction_character} ({confidence:.1f}%)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Show the annotated frame in a window titled “frame”
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()