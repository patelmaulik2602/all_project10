# import streamlit as st
# import time
# from docling.datamodel.base_models import InputFormat
# from docling.document_extractor import DocumentExtractor
#
# st.title("📄 Simple Invoice Extractor (Docling)")
#
# # -----------------------------
# # Upload File
# # -----------------------------
# uploaded_file = st.file_uploader(
#     "Upload PDF / PNG / JPG invoice",
#     type=["pdf", "png", "jpg", "jpeg"]
# )
#
# # -----------------------------
# # Extraction Template
# # -----------------------------
# template = """
# {
#     "Company":"string",
#     "Invoice":"string",
#     "Billing Address":"string",
#     "Shipping Address":"string",
#     "Shipping Method":"string",
#     "Order Number":"text",
#     "Order Date":"text",
#     "Payment Method":"string",
#     "Email":"string",
#     "Telephone/Phone":"string",
#     "Delivery-Date":"text",
#     "Delivery-Time":"text",
#     "products":[
#         {
#             "SKU":"text",
#             "Product":"string",
#             "Quantity":"text",
#             "Price":"string",
#             "Total":"string"
#         }
#     ],
#     "Subtotal":"string",
#     "Discount":"string",
#     "Shipping":"string",
#     "Refund":"string",
#     "Coupon-Used":"string",
#     "Total":"string"
# }
# """
#
# # -----------------------------
# # Docling Extractor
# # -----------------------------
# extractor = DocumentExtractor(
#     allowed_formats=[InputFormat.IMAGE, InputFormat.PDF]
# )
#
# # -----------------------------
# # Run Extraction
# # -----------------------------
# if uploaded_file:
#
#     # Save uploaded file temporarily
#     file_path = uploaded_file.name
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#
#     if st.button("Extract Data"):
#
#         with st.spinner("Extracting... please wait ⏳"):
#             start = time.time()
#
#             result = extractor.extract(
#                 source=file_path,
#                 template=template
#             )
#
#             end = time.time()
#
#         st.success("Done! 🎉")
#         st.json(result.pages)
#         st.info(f"Extraction time: {end - start:.2f} sec")

import streamlit as st
import time
import json
import os
import tempfile
import cv2
from typing import Optional
from docling.datamodel.base_models import InputFormat
from docling.document_extractor import DocumentExtractor
import requests

# ---------------------------
# LOGO CONFIGURATION
# ---------------------------
LOGO_FOLDER = r"C:\Users\Admin\Desktop\A_Maulik\Extract_Invoice\logo"
LOGO_MAP = {
    "Woo_Commerce.png": "WooCommerce",
    "Bresnans.png": "Bresnan's Family Butchers",
    "Company_Name.png": "CompanyName",
    "Addify.png": "Addify",
    "Lucky_Goldstar":"LuckyGoldstar"
}

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def load_and_prepare(img_path: str) -> Optional[cv2.Mat]:
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def crop_logo_area(img: cv2.Mat) -> cv2.Mat:
    h, w = img.shape[:2]
    return img[0:int(h * 0.25), 0:w]  # top 25% height

def detect_logo(img_path: str) -> Optional[str]:
    query_img = load_and_prepare(img_path)
    if query_img is None:
        return None
    query_crop = crop_logo_area(query_img)

    orb = cv2.ORB_create(600)
    kp1, des1 = orb.detectAndCompute(query_crop, None)
    if des1 is None:
        return None

    best_score = 0
    best_company = None

    for fname in os.listdir(LOGO_FOLDER):
        logo_path = os.path.join(LOGO_FOLDER, fname)
        if not os.path.isfile(logo_path):
            continue

        logo_img = load_and_prepare(logo_path)
        if logo_img is None:
            continue

        kp2, des2 = orb.detectAndCompute(logo_img, None)
        if des2 is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

        score = len(good) / max(1, min(len(kp1), len(kp2)))
        if score > best_score and score > 0.05:
            best_score = score
            best_company = LOGO_MAP.get(fname, os.path.splitext(fname)[0])
    return best_company

# ---------------------------
# CLEANING FUNCTION
# ---------------------------
def clean_text(val):
    if val is None:
        return ""
    if hasattr(val, "value") and not hasattr(val, "values"):
        return str(val.value).replace("*","").strip()
    if hasattr(val, "values"):
        return "\n".join([str(v.value).replace("*","").strip() if hasattr(v,"value") else str(v).replace("*","").strip() for v in val.values])
    return str(val).replace("*","").strip()

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("📄 Invoice Extractor (Docling + Logo Detection)")

uploaded_file = st.file_uploader("Upload PDF / PNG / JPG invoice", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

    # ---------------------------
    # Detect logo
    # ---------------------------
    company_name = detect_logo(temp_path)

    # ---------------------------
    # Docling Extraction Template
    # ---------------------------
    template = """
    {
        "Company":"string",
        "Invoice":"string",
        "Billing Address":"string",
        "Shipping Address":"string",
        "Shipping Method":"string",
        "Order Number":"text",
        "Order Date":"text",
        "Payment Method":"string",
        "Email":"string",
        "Telephone/Phone":"string",
        "Delivery-Date":"text",
        "Delivery-Time":"text",
        "products":[
            {
                "SKU":"text",
                "Product":"string",
                "Quantity":"text",
                "Price":"string",
                "Total":"string"
            }
        ],
        "Subtotal":"string",
        "Discount":"string",
        "Shipping":"string",
        "Refund":"string",
        "Coupon-Used":"string",
        "Total":"string"
    }
    """

    extractor = DocumentExtractor(allowed_formats=[InputFormat.IMAGE, InputFormat.PDF])

    if st.button("Extract Data"):
        with st.spinner("Extracting... please wait ⏳"):
            start = time.time()
            result = extractor.extract(source=temp_path, template=template)
            end = time.time()

        st.success("Extraction done! 🎉")

        # ---------------------------
        # Build clean JSON output
        # ---------------------------
        clean_output = {"company_name": company_name if company_name else ""}
        if result.pages:
            page_data = result.pages[0].extracted_data
            for key, val in page_data.items():
                clean_output[key] = val if val is not None else ""

        st.subheader("📄 Extracted Invoice Data")
        st.json(clean_output)
        st.info(f"Extraction time: {end - start:.2f} sec")

        # ---------------------------
        # Send to Django API
        # ---------------------------
        # ---------------------------
        # Send to Django API (FIXED like Mindee code)
        # ---------------------------
        DJANGO_API_URL = "http://127.0.0.1:8000/save-invoice/"
        if st.button("💾 Save to Database"):
            response = requests.post(DJANGO_API_URL, json=clean_output)
            if response.status_code == 201:
                st.success("Invoice saved successfully!")
            else:
                st.error("Failed to save invoice data!")

        # ---------------------------
        # DOWNLOAD JSON
        # ---------------------------
        json_bytes = json.dumps(clean_output, indent=4, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "⬇️ Download JSON",
            json_bytes,
            "invoice.json",
            "application/json"
        )