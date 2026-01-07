import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("📸 Image Detection")

# Load YOLO model
model = YOLO("yolo11x.pt")

# File uploader
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Photo", use_column_width=True)
    # # Run YOLO detection
    results = model(image)

    # Plot results (returns BGR numpy array)
    plotted_img = results[0].plot()

    # Convert BGR (OpenCV format) → RGB (PIL format)
    detected_image = Image.fromarray(plotted_img[..., ::-1])

    # Show detected image
    st.image(detected_image,caption="Detected Objects", use_column_width=True)

    st.success("✅ Detection complete!")
