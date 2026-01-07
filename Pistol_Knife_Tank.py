import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import cv2
import tempfile
from PIL import Image
import os

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = Path(r"C:\Users\Admin\Desktop\A_Maulik\Pistol_Knife_Detection\runs\Pistol_Knife_Tank_train\weights\best.pt")

# -------------------------------
# APP TITLE
# -------------------------------
st.set_page_config(page_title="🔫 Weapon Detection", layout="centered")
st.title("🔫 Weapon / Gun Detection System")
st.markdown("Upload an image below to detect weapons using your trained YOLO model.")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Keep file extension
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Display uploaded image
    image = Image.open(temp_path)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # -------------------------------
    # RUN DETECTION
    # -------------------------------
    st.write("⚙️ Running detection, please wait...")
    results = model.predict(source=temp_path, conf=0.25, save=False)

    # -------------------------------
    # DISPLAY RESULTS
    # -------------------------------
    result_img = results[0].plot()
    st.image(result_img, caption="🎯 Detection Result", use_container_width=True)

    # Print detection summary
    if len(results[0].boxes) > 0:
        st.success("✅ Weapon(s) detected!")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"🔹 **{model.names[cls]}** — Confidence: {conf:.2f}")
    else:
        st.warning("⚠️ No weapon detected in this image.")
else:
    st.info("👆 Upload an image to start detection.")
