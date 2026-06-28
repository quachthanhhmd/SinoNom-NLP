import google.generativeai as genai
from PIL import Image
from core.interfaces import OCRProvider

import base64
import requests

class GoogleVisionOCR(OCRProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            print("[Warning] GOOGLE_VISION_API_KEY is not set. Google Vision OCR will mock.")
            
    def extract_text(self, image_path: str) -> str:
        if not self.api_key:
            return "Tự Hán (Google Vision - No API Key)"
            
        print(f"[GoogleVisionOCR] Extracting text from {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                content = base64.b64encode(image_file.read()).decode('utf-8')
                
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
            
            payload = {
                "requests": [
                    {
                        "image": {
                            "content": content
                        },
                        "features": [
                            {
                                "type": "DOCUMENT_TEXT_DETECTION"
                            }
                        ],
                        "imageContext": {
                            "languageHints": ["zh-Hant", "lzh"]
                        }
                    }
                ]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            result = response.json()
            
            if response.status_code != 200 or "error" in result:
                err_msg = result.get("error", {}).get("message", "Unknown Error")
                return f"Tự Hán (Google Vision - API Error: {err_msg})"
                
            responses = result.get("responses", [])
            if not responses or not responses[0].get("fullTextAnnotation"):
                return ""
                
            # Vision natively detects vertical text layout often.
            return responses[0]["fullTextAnnotation"]["text"].strip()
            
        except Exception as e:
            import traceback
            print(f"[GoogleVisionOCR] Error: {e}")
            traceback.print_exc()
            return "Tự Hán (Google Vision - Error)"

class KanDianGuJiOCR(OCRProvider):
    def __init__(self, token: str = "", email: str = ""):
        self.token = token
        self.email = email
        self.api_url = "https://ocr.kandianguji.com/ocr_api"
        
    def extract_text(self, image_path: str) -> str:
        if not self.token or not self.email:
            print("[Warning] KanDianGuJi Token or Email not set. Skipping.")
            return "Tự Hán (KanDianGuJi - No Token)"
            
        print(f"[KanDianGuJiOCR] Extracting text from {image_path}")
        try:
            import requests
            import base64
            
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                
            payload = {
                "token": self.token,
                "email": self.email,
                "image": image_base64,
                "version": "v2",
                "det_mode": "sp"  # sp: 竖排 (vertical), hp: 横排 (horizontal)
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse KanDianGuJi JSON response structure
                # Typically it returns {"code": 0, "msg": "success", "data": [...]}
                # The "data" field usually contains bounding boxes and texts.
                # Assuming the API returns a list of text segments inside "data"
                
                if result.get("code") == 0 or result.get("msg") == "success":
                    data = result.get("data", [])
                    
                    text_lines = []
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            text_lines.append(item["text"])
                        elif isinstance(item, str):
                            text_lines.append(item)
                            
                    return "\n".join(text_lines)
                else:
                    print(f"[KanDianGuJiOCR] API Error: {result.get('msg', 'Unknown Error')}")
                    return f"Tự Hán (KanDianGuJi - {result.get('msg', 'API Error')})"
            else:
                print(f"[KanDianGuJiOCR] HTTP Error: {response.status_code}")
                return "Tự Hán (KanDianGuJi - HTTP Error)"
                
        except Exception as e:
            print(f"[KanDianGuJiOCR] Error: {e}")
            return "Tự Hán (KanDianGuJi - Error)"

class PaddleOCRProvider(OCRProvider):
    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            # lang='chinese_cht' for Traditional Chinese (Sino-Nom)
            # use_angle_cls=True helps correct any residual tilt after rotation
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='chinese_cht',
                det_db_unclip_ratio=1.8,  # Expand boxes slightly to catch cut off chars at the end
                det_db_box_thresh=0.4     # Lower threshold to keep faint/thin lines
            )
        except ImportError:
            print("[Warning] PaddleOCR is not installed.")
            self.ocr = None

    def extract_text(self, image_path: str) -> str:
        if not self.ocr:
            return "Tự Hán (Paddle - Not Installed)"

        print(f"[PaddleOCR] Extracting text from {image_path}")

        try:
            import cv2
            import re

            # ── Pre-processing: rotate 90° counter-clockwise ──────────────────
            # Original document: vertical columns, read RIGHT → LEFT, TOP → BOTTOM
            # After CCW rotation: center_x = y_orig, center_y = W_orig - x_orig
            #   • Province name (rightmost in original = large x_orig) → small center_y
            #   • Province band (y position range in original) → center_x range
            image = cv2.imread(image_path)
            if image is None:
                return "Tự Hán (Paddle - Image Load Error)"

            # 1. Padding: add a 50px white border to prevent edge characters from being cropped
            image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            # 2. Grayscale (No extreme binarization to preserve calligraphy nuances)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # PaddleOCR expects a 3-channel image for its internal pipeline
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ── OCR ───────────────────────────────────────────────────────────
            result = self.ocr.ocr(rotated)

            if not result or not result[0]:
                return ""

            # PaddleOCR 2.8.x returns: [ [box_coords, (text, confidence)], ... ]
            # where box_coords = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            ocr_result = result[0]

            # ── Post-processing ───────────────────────────────────────────────
            boxes = []
            for line in ocr_result:
                poly = line[0]           # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = line[1][0]        # recognized text
                conf = line[1][1]        # confidence score

                if conf < 0.4:
                    continue
                # Filter out pure ASCII/noise
                if re.fullmatch(r'[A-Za-z0-9\s\W]+', text):
                    continue
                center_y = sum(pt[1] for pt in poly) / len(poly)
                center_x = sum(pt[0] for pt in poly) / len(poly)
                boxes.append((center_y, center_x, text))

            # ── Post-processing: column-band grouping ─────────────────────────
            # After CCW rotation: center_x = y_orig (province data band position)
            # Each province occupies a BAND of center_x values (its y position range)
            # Sort boxes by center_x ascending, then detect gaps to form bands.
            # Bands are sorted descending by avg center_x (底部→總計 first, 頂部→慶和省 last).
            # Within each band, sort by center_y ascending (province name = rightmost
            # in original = smallest center_y → appears first). ✓

            boxes.sort(key=lambda b: b[1])  # sort by center_x ascending

            # Gap-based band detection: split when gap > BAND_THRESHOLD
            BAND_THRESHOLD = 90  # pixels between consecutive center_x values
            bands = []
            for box in boxes:
                if not bands:
                    bands.append([box])
                else:
                    gap = box[1] - bands[-1][-1][1]
                    if gap > BAND_THRESHOLD:
                        bands.append([box])
                    else:
                        bands[-1].append(box)

            # Sort bands by average center_x DESCENDING
            # (large center_x = large y_orig = bottom of page = 總計 section first)
            bands.sort(key=lambda band: sum(b[1] for b in band) / len(band), reverse=True)

            # Within each band, sort by center_y ASCENDING
            # (small center_y = large x_orig = rightmost in original = province name first)
            text_lines = []
            for band in bands:
                band.sort(key=lambda b: b[0])
                for box in band:
                    text_lines.append(box[2])


            # ── Post-corrections ──────────────────────────────────────────────
            corrections = {
                "廣義脊": "廣義省",
                "四土":   "田土",
                "五萬空百": "五萬八百",
                "二十九部": "二十九畝",
                "稅案":   "稅粟",
            }
            corrected = []
            for line in text_lines:
                for wrong, right in corrections.items():
                    line = line.replace(wrong, right)
                corrected.append(line)

            return "\n".join(corrected)

        except Exception as e:
            import traceback
            print(f"[PaddleOCR] Error: {e}")
            traceback.print_exc()
            return "Tự Hán (Paddle - Error)"


