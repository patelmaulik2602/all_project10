import streamlit as st
import os
from ultralytics import YOLO

st.set_page_config(page_title="Person Detection", layout="centered")

# Title
st.title("🧍 Person Detection in Video")

# Upload section
uploaded_video = st.file_uploader("📤 Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save uploaded video temporarily
    input_video_path = os.path.join("input_videos", uploaded_video.name)
    os.makedirs("input_videos", exist_ok=True)
    with open(input_video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(input_video_path)
    st.success("✅ Video uploaded successfully!")

    # Run detection button
    if st.button("🚀 Run Person Detection"):
        with st.spinner("Running detection... Please wait ⏳"):
            # Load YOLO model
            model = YOLO("yolov9s.pt")  # You can use yolov8n.pt or any YOLOv8/v9 variant

            # Set output path (Desktop)
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            project_path = os.path.join(desktop_path, "YOLO_Person_Detection_Output")

            # Run detection
            results = model.predict(
                source=input_video_path,
                save=True,
                conf=0.5,
                show=False,
                project=project_path,
                name="Result",
                classes=[0]  # Only detect persons (class 0)
            )

            # Path to output video
            output_video_path = os.path.join(project_path, "Result", uploaded_video.name)

            # Show result
            st.success("✅ Detection complete! Saved on your Desktop:")
            st.code(output_video_path, language="bash")

            if os.path.exists(output_video_path):
                st.video(output_video_path)
            else:
                st.error("⚠️ Could not find output video. Please check the Desktop folder.")
