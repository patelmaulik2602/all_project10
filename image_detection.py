import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import timm
import torchvision.transforms as T
import numpy as np
import urllib.request

st.title("🖼️ Image Detection")

# -----------------------
# 1. Load Models
# -----------------------
# YOLO detector (COCO dataset = 80 classes)
detector = YOLO("yolo11m.pt")  # use yolo11l.pt or yolo11x.pt for better accuracy

# ImageNet classifier (ResNet50)
classifier = timm.create_model("resnet152", pretrained=True)
classifier.eval()

# ✅ Load ImageNet labels manually
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# Transform for classifier
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -----------------------
# 2. File Uploader
# -----------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",width=300)

    # -----------------------
    # 3. Run YOLO Detection
    # -----------------------
    results = detector(image, conf=0.3)  # lower conf → more detections
    img_array = np.array(image)

    st.subheader("🔎 Detected Objects")
    for box in results[0].boxes:
        # YOLO Prediction
        cls_id = int(box.cls[0].item())
        yolo_label = results[0].names[cls_id]
        yolo_conf = float(box.conf[0].item())

        # Extract bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_array[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop)

        # -----------------------
        # 4. Classify with ResNet
        # -----------------------
        input_tensor = transform(crop_pil).unsqueeze(0)
        with torch.no_grad():
            preds = classifier(input_tensor)
            cls_id_imgnet = preds.argmax(dim=1).item()
            cls_label = imagenet_labels[cls_id_imgnet]
            cls_conf = torch.nn.functional.softmax(preds, dim=1)[0, cls_id_imgnet].item()

        # -----------------------
        # 5. Display Results
        # -----------------------

        st.image(
            crop_pil,
            caption=(

                f"YOLO: {yolo_label} ({yolo_conf:.2f})\n"
                f"ResNet: {cls_label} ({cls_conf:.2f})"

            ),
            width=300

        )

    st.success("✅ Detection Completed!")


