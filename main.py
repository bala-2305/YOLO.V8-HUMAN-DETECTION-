import streamlit as st
import cv2
from detector import HumanAnalytics

st.set_page_config(layout="wide")
st.title("🧠 Human Intelligence Detection")

start_btn = st.button("Start Webcam")
stop_btn = st.button("Stop")

frame_placeholder = st.empty()
log_placeholder = st.empty()

if start_btn:
    system = HumanAnalytics(source=0)
    system.run(
        frame_callback=frame_placeholder.image,
        log_callback=log_placeholder.text
    )

if stop_btn:
    st.warning("Stopped. Refresh to restart.")
