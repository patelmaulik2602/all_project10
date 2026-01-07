import os
from glob import glob
from PIL import Image
from ultralytics import YOLO

# Folder containing images
folder_path = r"C:\Users\Admin\Desktop\Images"

# Get all images (jpg/png)
image_files = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))

if not image_files:
    print("No images found in the folder.")
else:
    # Show all images with original filenames
    print("Images in folder:")
    for i, img_path in enumerate(image_files, start=1):
        print(f"{i}: {os.path.basename(img_path)}")

# Choose image
choice = int(input(f"Enter the number of the image to process (1-{len(image_files)}): "))

if 1 <= choice <= len(image_files):
    img_path = image_files[choice - 1]  # adjust index
    print("Processing:", img_path)

    # Load image
    image = Image.open(img_path)

    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Run model and save result
    result = model(image, save=True)

    # Show result
    result[0].show()
else:
    print("Invalid number.")
