import cv2
import numpy as np
import fitz
import os
from ocr.ocr_pipeline import detect_columns, preprocess
import sys

doc = fitz.open("dataset/china/han_pdf/02.pdf")
page = doc.load_page(6)
pix = page.get_pixmap(dpi=300)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

page_img, thresh = preprocess(None, img, "debug_output")
print(f"Page shape: {page_img.shape}")
columns_info = detect_columns(page_img, thresh, "debug_output")
print(f"Detected {len(columns_info)} columns")
for c in columns_info:
    print(f"x_left: {c['x_left']}, x_right: {c['x_right']}, width: {c['width']}")
