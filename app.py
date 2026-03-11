import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from collections import deque

st.title("🖐 Real-Time Sign Language Interpreter")

# Load trained ML model
model = joblib.load("gesture_model.pkl")

# Load MediaPipe Hand Landmarker
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

# Prediction smoothing
predictions = deque(maxlen=10)

start = st.button("Start Camera")

frame_placeholder = st.image([])

if start:

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            st.write("Camera error")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                landmarks = []

                for lm in hand:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                if len(landmarks) == 63:

                    prediction = model.predict([landmarks])[0]

                    predictions.append(prediction)

                    final_prediction = max(set(predictions), key=predictions.count)

                    cv2.putText(
                        frame,
                        final_prediction.upper(),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2
                    )

        frame_placeholder.image(frame, channels="BGR")

    cap.release()