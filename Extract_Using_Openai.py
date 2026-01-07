import streamlit as st
import json
import tempfile
import os
from openai import OpenAI
from PIL import Image
import pytesseract
import re
import cv2
import numpy as np
import fitz  # PyMuPDF

# ---------------------------
# OpenAI Client
# ---------------------------
client = OpenAI(api_key="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")  # Replace with your OpenAI API key
# client = OpenAI(api_key="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")

# Windows example: replace with your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



INVOICE_SCHEMA = """
You are an invoice parser. Convert the invoice text into clean JSON.
Ensure every field exists: use empty string "" or 0 if missing.


JSON FORMAT:
{
  
  "invoice": "string",
  "billing_address": "string",
  "shipping_address": "string",
  "order_number": "string",
  "order_date": "string",
  "payment_method": "string",
  "email": "string",
  "phone": "string",
  "delivery_date": "string",
  "delivery_time": "string",
  "subtotal": "number",
  "discount": "number",
  "shipping": "number",
  "refund": "number",
  "total": "number",
  "coupon_used": "string",
  "payment_items": [
    {
      "sku": "string",
      "product": "string",
      "quantity": "number",
      "price": "number"
    }
  ]
}
"""

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_image(image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    # Adaptive thresholding works well for scanned invoices
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    return Image.fromarray(thresh)

def extract_text_ocr(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            preprocessed = preprocess_image(img)
            text += pytesseract.image_to_string(preprocessed, lang="eng") + "\n"
    else:
        preprocessed = preprocess_image(Image.open(file_path).convert("RGB"))
        text = pytesseract.image_to_string(preprocessed, lang="eng")
    return text

def clean_text(text):
    text = text.replace("\n\n", "\n")
    text = re.sub(r"\s{2,}", " ", text)
    return text

def extract_invoice_json(text):
    for _ in range(2):
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": INVOICE_SCHEMA},
                {"role": "user", "content": text},
            ]
        )
        output = completion.choices[0].message.content
        try:
            return json.loads(output)
        except:
            text = output
    return {}

def ensure_defaults(parsed_json):
    defaults = {
                 "invoice": "",
                 "billing_address": "",
                 "shipping_address": "",
                 "order_number": "",
                 "order_date": "",
                 "payment_method": "",
                 "email": "",
                 "phone": "",

                 "delivery_time": "",
                 "subtotal": 0,
                 "discount": 0,
                 "shipping": 0,
                 "refund": 0,
                 "total": 0,
                 "coupon_used": "",
                 "payment_items": []

    }
    for key, value in defaults.items():
        if key not in parsed_json or parsed_json[key] is None:
            parsed_json[key] = value

    items = parsed_json.get("payment_items", [])
    for item in items:
        print(parsed_json)
        item["sku"] = str(item.get("sku", ""))
        item["product"] = str(item.get("product", ""))
        item["quantity"] = str(item.get("quantity", ""))
        item["price"] = str(item.get("price", ""))
    parsed_json["payment_items"] = items
    return parsed_json

def process_invoice(file_path):
    raw_text = extract_text_ocr(file_path)
    if not raw_text.strip():
        return {"error": "Could not extract text from this invoice. Try a clearer PDF or image."}
    raw_text = clean_text(raw_text)
    parsed = extract_invoice_json(raw_text)
    parsed = ensure_defaults(parsed)
    return parsed

# ---------------------------
# Streamlit App
# ---------------------------
st.title("Invoice Extraction Without Poppler (OCR + GPT)")

uploaded_file = st.file_uploader("Upload PDF or Image Invoice", type=["pdf", "png", "jpg", "jpeg"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.info("Processing invoice... please wait")
    try:
        result = process_invoice(tmp_file_path)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Invoice processed successfully!")
            st.json(result)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        os.unlink(tmp_file_path)
