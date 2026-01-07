# import streamlit as st
# import json
# import requests
# from mindee import ClientV2, InferenceParameters, PathInput
# import tempfile
# import os
# import re
#
# st.title("🧾 Invoice Extraction using Mindee + Streamlit")
#
# uploaded_file = st.file_uploader("Upload Invoice Image/PDF", type=["png", "jpg", "jpeg", "pdf"])
#
# if uploaded_file:
#     st.success("File uploaded successfully!")
#
#     # Save temp file
#     temp_dir = tempfile.mkdtemp()
#     temp_path = os.path.join(temp_dir, uploaded_file.name)
#
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.read())
#
#     # Mindee Setup
#     api_key = "md_Oyvbp74v7wkBsH1mTOzyvZzAdKbilWKT"
#     model_id = "4c897a4f-0d0e-4018-b39a-95fc2061625e"
#
#     mindee_client = ClientV2(api_key)
#     params = InferenceParameters(model_id=model_id)
#     input_source = PathInput(temp_path)
#
#     # Extract data
#     with st.spinner("Extracting data using Mindee..."):
#         response = mindee_client.enqueue_and_get_inference(input_source, params)
#
#     fields = response.inference.result.fields
#
#
#     # ---------------------------------------------------------
#     # CLEANING FUNCTIONS
#     # ---------------------------------------------------------
#     def clean_line(text):
#         if not text:
#             return ""
#         return text.replace("*", "").strip()
#
#
#     def clean_text(val):
#         if val is None:
#             return ""
#
#         if hasattr(val, "value") and not hasattr(val, "values"):
#             return clean_line(str(val.value))
#
#         if hasattr(val, "values"):
#             cleaned = []
#             for v in val.values:
#                 if hasattr(v, "value"):
#                     cleaned.append(clean_line(str(v.value)))
#                 else:
#                     cleaned.append(clean_line(str(v)))
#             return "\n".join([x for x in cleaned if x])
#
#         return clean_line(str(val))
#
#
#     # ---------------------------------------------------------
#     # PAYMENT DETAILS PARSER
#     # ---------------------------------------------------------
#     def parse_payment_details(text):
#         items = []
#         blocks = re.split(r"\s*:sku:\s*", text, flags=re.I)
#
#         for block in blocks:
#             block = block.strip()
#             if not block:
#                 continue
#
#             lines = block.split("\n")
#             sku = lines[0].strip()
#
#             product_match = re.search(r":product:\s*([\s\S]*?):quantity:", block, re.I)
#             quantity_match = re.search(r":quantity:\s*([\d.]+)", block, re.I)
#             price_match = re.search(r":price:\s*([\d.]+)", block, re.I)
#
#             product = ""
#             if product_match:
#                 product = (
#                     product_match.group(1)
#                     .strip()
#                     .replace("\n", " ")
#                     .replace("  ", " ")
#                 )
#
#             items.append({
#                 "sku": sku,
#                 "product": product,
#                 "quantity": float(quantity_match.group(1)) if quantity_match else 0,
#                 "price": float(price_match.group(1)) if price_match else 0
#             })
#
#         return items
#
#
#     # ---------------------------------------------------------
#     # BUILD CLEAN OUTPUT
#     # ---------------------------------------------------------
#     clean_output = {}
#     for key, val in fields.items():
#         clean_output[key] = clean_text(val)
#
#     # Extract payment items if present
#     payment_details_raw = clean_output.get("payment_details", "")
#
#     if payment_details_raw:
#         clean_output["payment_items"] = parse_payment_details(payment_details_raw)
#     else:
#         clean_output["payment_items"] = []
#
#     # ---------------------------------------------------------
#     # SHOW JSON OUTPUT
#     # ---------------------------------------------------------
#     st.subheader("📄 Extracted Invoice Data")
#     st.json(clean_output)
#
#     # ---------------------------------------------------------
#     # SEND TO DJANGO
#     # ---------------------------------------------------------
#     DJANGO_API_URL = "http://127.0.0.1:8000/save-invoice/"
#
#     if st.button("💾 Save to Database"):
#         response = requests.post(DJANGO_API_URL, json=clean_output)
#
#         if response.status_code == 201:
#             st.success("Invoice saved successfully!")
#         else:
#             st.error("Failed to save invoice data!")
#
#     # ---------------------------------------------------------
#     # DOWNLOAD JSON
#     # ---------------------------------------------------------
#     json_bytes = json.dumps(clean_output, indent=4, ensure_ascii=False).encode("utf-8")
#
#     st.download_button(
#         "⬇️ Download JSON",
#         json_bytes,
#         "invoice.json",
#         "application/json"
#     )




import streamlit as st
import json
import requests
from mindee import ClientV2, InferenceParameters, PathInput
import tempfile
import os
import re
import cv2
import fitz  # PyMuPDF
from typing import Optional

# ---------------------------
# LOGO CONFIGURATION
# ---------------------------
LOGO_FOLDER = r"C:\Users\Admin\Desktop\A_Maulik\Extract_Invoice\logo"  # folder where your logo images are stored
LOGO_MAP = {
    "Woo_Commerce.png": "WooCommerce",
    "Bresnans.png": "Bresnan's",
    "Company_Name.png": "CompanyName",
    "Addify.png": "Addify"
}
# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def pdf_to_image(pdf_path: str, page_number=0, dpi=300) -> str:
    """Converts the first page of PDF to an image and returns the path."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    img_path = f"{os.path.splitext(pdf_path)[0]}_page{page_number}.png"
    pix.save(img_path)
    return img_path

def crop_logo_area(img: cv2.Mat) -> cv2.Mat:
    """Crop the top region of the image where logos usually are (full width)."""
    h, w = img.shape[:2]
    return img[0:int(h * 0.25), 0:w]  # top 25% height, full width

def load_and_prepare(img_path: str) -> Optional[cv2.Mat]:
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def detect_logo(img_path: str, logo_folder=LOGO_FOLDER, logo_map=LOGO_MAP) -> Optional[str]:
    """Detects the company logo from an image using ORB feature matching."""
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

    for fname in os.listdir(logo_folder):
        logo_path = os.path.join(logo_folder, fname)
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
        if score > best_score and score > 0.05:  # threshold
            best_score = score
            best_company = logo_map.get(fname, os.path.splitext(fname)[0])

    return best_company

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🧾 Invoice Extraction with Mindee + Logo Detection")

uploaded_file = st.file_uploader("Upload Invoice Image/PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    st.success("File uploaded successfully!")

    # Save temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # ---------------------------
    # Logo Detection
    # ---------------------------
    ext = os.path.splitext(temp_path)[1].lower()
    if ext == ".pdf":
        img_path = pdf_to_image(temp_path)
    else:
        img_path = temp_path

    company_name = detect_logo(img_path)
    # if company_name:
    #     st.success(f"Detected Logo: {company_name}")
    #     st.image(img_path, caption="Uploaded / Extracted Image", use_container_width=True)
    # else:
    #     st.warning("No logo detected or match not found.")
    #     st.image(img_path, caption="Uploaded / Extracted Image", use_container_width=True)

    # ---------------------------
    # Mindee Setup
    # ---------------------------
    api_key = "md_Oyvbp74v7wkBsH1mTOzyvZzAdKbilWKT"
    model_id = "4c897a4f-0d0e-4018-b39a-95fc2061625e"



    mindee_client = ClientV2(api_key)
    params = InferenceParameters(model_id=model_id)
    input_source = PathInput(temp_path)

    # Extract data
    with st.spinner("Extracting data using Mindee..."):
        response = mindee_client.enqueue_and_get_inference(input_source, params)

    fields = response.inference.result.fields

    # ---------------------------
    # CLEANING FUNCTIONS
    # ---------------------------
    def clean_line(text):
        if not text:
            return ""
        return text.replace("*", "").strip()

    def clean_text(val):
        if val is None:
            return ""
        if hasattr(val, "value") and not hasattr(val, "values"):
            return clean_line(str(val.value))
        if hasattr(val, "values"):
            cleaned = []
            for v in val.values:
                if hasattr(v, "value"):
                    cleaned.append(clean_line(str(v.value)))
                else:
                    cleaned.append(clean_line(str(v)))
            return "\n".join([x for x in cleaned if x])
        return clean_line(str(val))

    # ---------------------------
    # PAYMENT DETAILS PARSER
    # ---------------------------
    # def parse_payment_details(text):
    #     items = []
    #     blocks = re.split(r"\s*:sku:\s*", text, flags=re.I)
    #     for block in blocks:
    #         block = block.strip()
    #         if not block:
    #             continue
    #         lines = block.split("\n")
    #         sku = lines[0].strip()
    #         product_match = re.search(r":product:\s*([\s\S]*?):quantity:", block, re.I)
    #         quantity_match = re.search(r":quantity:\s*([\d.]+)", block, re.I)
    #         price_match = re.search(r":price:\s*([\d.]+)", block, re.I)
    #         product = ""
    #         if product_match:
    #             product = product_match.group(1).strip().replace("\n", " ").replace("  ", " ")
    #         items.append({
    #             "sku": sku,
    #             "product": product,
    #             "quantity": float(quantity_match.group(1)) if quantity_match else 0,
    #             "price": float(price_match.group(1)) if price_match else 0
    #         })
    #     return items

    import re


    def parse_payment_details(text):
        items = []

        # Split only when real :sku: exists
        blocks = re.split(r"\s*:sku:\s*", text, flags=re.I)

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            # Only allow SKU if real :sku: tag existed
            sku = ""
            if i != 0 and not block.lower().startswith(":product:"):
                sku = block.split("\n")[0].strip()

            product_match = re.search(r":product:\s*(.*?):quantity:", block, re.I | re.S)
            quantity_match = re.search(r":quantity:\s*([\d.]+)", block, re.I)
            price_match = re.search(r":price:\s*([\d.]+)", block, re.I)

            product = ""
            if product_match:
                product = product_match.group(1).strip().replace("\n", " ")

            items.append({
                "sku": sku,  # ✅ will be "" if not found
                "product": product,  # ✅ correct product
                "quantity": float(quantity_match.group(1)) if quantity_match else 0,
                "price": float(price_match.group(1)) if price_match else 0
            })

        return items


    # ---------------------------
    # BUILD CLEAN OUTPUT
    # ---------------------------
    # ---------------------------
    # BUILD CLEAN OUTPUT
    # ---------------------------
    # Start with company name first
    clean_output = {"company_name": company_name if company_name else ""}

    # Then add all other extracted fields
    for key, val in fields.items():
        clean_output[key] = clean_text(val)

    # Extract payment items if present
    payment_details_raw = clean_output.get("payment_details", "")
    if payment_details_raw:
        clean_output["payment_items"] = parse_payment_details(payment_details_raw)
    else:
        clean_output["payment_items"] = []

    # ---------------------------
    # SHOW JSON OUTPUT
    # ---------------------------
    st.subheader("📄 Extracted Invoice Data")
    st.json(clean_output)

    # ---------------------------
    # SEND TO DJANGO
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
