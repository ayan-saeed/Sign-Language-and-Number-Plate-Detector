# Sign-Language-Detector
## Overview
Real-time Sign Language Interpreter using Python, and Scikit-learn. Captures webcam input, detects hand signs, and classifies gestures into sign language symbols using a trained ML model.

## Initial Requirements 
1. Video Capture - The system must continuously capture real-time video frames from the webcam
2. Hand Detection - The system must detect the presence of one or more hands in the video frame
3. Hand Sign Extraction - The system must extract and process key hand landmarks from each detected hand to represent the hand’s position, orientation, and finger joint configuration.
4. Gesture Classification - The system must classify hand gestures into predefined gesture labels using a trained Scikit-learn model.
5. Prediction Display - The system must overlay the predicted gesture label onto the live video feed above the detected hand.

## Further Requirements
6. Implement into a Web Application 