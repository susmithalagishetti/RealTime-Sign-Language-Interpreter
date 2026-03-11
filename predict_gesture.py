import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyttsx3

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from collections import deque

# Load trained ML model
model = joblib.load("gesture_model.pkl")

# Text-to-speech engine
engine = pyttsx3.init()
last_spoken = ""

# Prediction smoothing buffer
predictions = deque(maxlen=10)

# Load MediaPipe Hand Landmarker
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect hand landmarks
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            landmarks = []

            for lm in hand:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            # Ensure correct landmark size
            if len(landmarks) == 63:

                prediction = model.predict([landmarks])[0]

                # Add to prediction buffer
                predictions.append(prediction)

                # Majority vote smoothing
                final_prediction = max(set(predictions), key=predictions.count)

                # Display text
                cv2.putText(
                    frame,
                    final_prediction.upper(),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Voice output
                if final_prediction != last_spoken:
                    engine.say(final_prediction)
                    engine.runAndWait()
                    last_spoken = final_prediction

    # Show webcam window
    cv2.imshow("Sign Language Interpreter", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()