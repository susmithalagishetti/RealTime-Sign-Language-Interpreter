import cv2
import mediapipe as mp
import csv

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

label = input("Enter gesture name: ")

with open("dataset.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:

        ret, frame = cap.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            landmarks = []

            for lm in result.hand_landmarks[0]:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks.append(label)

            writer.writerow(landmarks)

            print("Sample saved")

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()