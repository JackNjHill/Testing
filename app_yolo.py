import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv11 model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Make sure this file exists in the repo
    return model

model = load_model()

# Reckless behavior class names
reckless_classes = ['Texting', 'Talking on the phone', 'Reaching Behind', 
                    'Hair and Makeup', 'Eyes Closed', 'Yawning', 'Nodding Off']

# Streamlit UI
st.title("🚗 Reckless Driving Detection with YOLOv11")
st.markdown("Upload a video and detect any signs of distracted or reckless driving behavior.")

uploaded_video = st.file_uploader("📹 Upload a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    # Show uploaded video
    st.video(video_path)

    cap = cv2.VideoCapture(video_path)
    placeholder = st.empty()

    reckless_detected = False
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(rgb_frame, verbose=False)[0]  # YOLOv11 returns list
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)

        flagged = False
        for i, cls_id in enumerate(class_ids):
            label = model.names[cls_id]
            if label in reckless_classes:
                flagged = True
                reckless_detected = True

                # Draw bounding box
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(rgb_frame, f"Reckless: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        caption = f"Frame {frame_num}: {'🚨 Reckless behavior detected!' if flagged else '✅ Clear'}"
        placeholder.image(rgb_frame, caption=caption, use_column_width=True, channels="RGB")

    cap.release()

    if reckless_detected:
        st.warning("⚠️ Reckless behavior detected in the video.")
    else:
        st.success("✅ No reckless behavior detected.")