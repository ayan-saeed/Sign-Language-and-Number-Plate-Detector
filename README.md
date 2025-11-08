# Sign Language and Number Plate Detector
## Sign Language Detector
### Overview
Real-time Sign Language Interpreter using Python, MediaPipe, and Scikit-learn. Captures webcam input, detects hand signs, and classifies gestures into sign language symbols using a trained ML model.

### Sign Language Detector Requirements 
1. Video Capture - The system must continuously capture real-time video frames from the webcam. (✅)
2. Hand Detection - The system must detect the presence of one or more hands in the video frame. (✅)
3. Hand Sign Extraction - The system must extract and process key hand landmarks from each detected hand to represent the hand’s position, orientation, and finger joint configuration. (✅)
4. Gesture Classification - The system must classify hand gestures into predefined gesture labels using a trained Scikit-learn model. (✅)
5. Prediction Display - The system must overlay the predicted gesture label onto the live video feed above the detected hand. (✅)

### Final Showcase
![Hand Sign - B](/Sign-Language-files/images/B.png)
![Hand Sign - E](/Sign-Language-files/images/E.png)

## Number Plate Detector
### Overview
Real-time Number Plate Detector using Python, OpenCV, and Optical Character Recognition (OCR). Captures images or video frames, detects and isolates vehicle license plates, enhances clarity, and extracts alphanumeric text using EasyOCR or Tesseract.

### Sign Language Detector Requirements
1. Video Capture — The system must continuously capture real-time video frames or process still images
2. Plate Detection — The system must detect and locate the number plate region within the frame
3. Region Extraction — The system must isolate the detected region from Requirement 2
4. Text Recognition — The system must extract alphanumeric text from the plate using an OCR engine such as EasyOCR or Tesseract
5. Result Display — The system must overlay the detected plate region and recognized text onto the live video feed, including the accuracy or confidence score of the prediction

### Final Showcase