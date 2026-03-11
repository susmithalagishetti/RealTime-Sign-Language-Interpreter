import cv2
import mediapipe as mp

# Initialize MediaPipe Hands from tasks API
mp_hands = mp.tasks.vision.HandLandmarker

print("MediaPipe installed correctly!")