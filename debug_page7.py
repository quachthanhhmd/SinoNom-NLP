import cv2
import numpy as np
import fitz
import os
from ocr.ocr_pipeline import ocr_sinonom_page
from paddleocr import PaddleOCR

print("Loading OCR model...")
ocr_engine = PaddleOCR(use_angle_cls=False, lang="ch", det_db_box_thresh=0.3, det_db_unclip_ratio=2.0)

print("Converting PDF...")
doc = fitz.open("dataset/china/han_pdf/02.pdf")
page = doc.load_page(6) # 0-indexed for page 7
pix = page.get_pixmap(dpi=300)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

print("Running pipeline...")
os.makedirs("debug_output", exist_ok=True)
results = ocr_sinonom_page(image_np=img, debug_ocr=True, output_dir="debug_output", ocr_engine=ocr_engine, verbose=True, is_half_page=True, pdf_page_num=7, pdf_filename="dataset/china/han_pdf/02.pdf")

print("RESULTS:")
for r in results:
    print(f"[{r['page_number']}] Middle: {r['middle']}, Text: {r['text']}, Bbox: {r['bbox']}")
