from ultralytics import YOLO
import cv2
from emotion_model import predict_emotion
from age_model import predict_age_group
from hand_sign import get_hand_signs
import numpy as np

class HumanAnalytics:
    def __init__(self, source=0):
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(source)
        self.person_id = 0

    def run(self, frame_callback=None, log_callback=None):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame)[0]
            people = [r for r in results.boxes.data.tolist() if int(r[5]) == 0]  # person class only

            for i, person in enumerate(people):
                x1, y1, x2, y2, conf, cls_id = map(int, person[:6])
                face = frame[y1:y2, x1:x2]
                
                # Emotion
                emotion = predict_emotion(face)

                # Age
                age_group = predict_age_group(face)

                # Hand signs
                hands = get_hand_signs(frame)

                label = f"Person {i+1} | {emotion} | {age_group}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            cv2.putText(frame, f"Total People: {len(people)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if frame_callback:
                frame_callback(frame, channels="BGR")

            if log_callback:
                log_callback(f"Detected: {len(people)} people | Hands: {hands}")

        self.cap.release()
