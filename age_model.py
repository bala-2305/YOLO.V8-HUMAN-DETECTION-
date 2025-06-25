import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/age_group_model.h5")
age_labels = ['Child', 'Teen', 'Adult', 'Middle Age', 'Senior']

def predict_age_group(face):
    try:
        resized = cv2.resize(face, (64, 64))
        img = resized.reshape(1, 64, 64, 3) / 255.0
        prediction = model.predict(img)[0]
        return age_labels[np.argmax(prediction)]
    except:
        return "Unknown"
