import mediapipe as mp
import cv2

hands_model = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
sign_labels = {
    "fist": "👊",
    "open": "✋",
    "thumbs_up": "👍",
    "thumbs_down": "👎",
    "peace": "✌️"
}

def get_hand_signs(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "No hands"

    signs = []
    for hand in results.multi_hand_landmarks:
        # Simple rule-based thumb detection (for demo)
        # Real deployment needs trained CNN
        signs.append("open")  # Placeholder

    return ", ".join(signs)
