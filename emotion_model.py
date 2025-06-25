import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/emotion_model.h5")
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face):
    try:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        img = resized.reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(img)[0]
        return emotion_labels[np.argmax(prediction)]
    except:
        return "Unknown"
