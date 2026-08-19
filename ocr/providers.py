import google.generativeai as genai
from PIL import Image
from core.interfaces import OCRProvider

import base64
import requests
import os

class GeminiOCRProvider(OCRProvider):
    def __init__(self):
        try:
            from config import Config
            key = Config.GEMINI_API_KEY
        except ImportError:
            key = None
            
        self.api_key = key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            print("[Warning] GEMINI_API_KEY is not set. Gemini OCR will not run.")
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model_name = 'gemini-1.5-flash'
            self.model = genai.GenerativeModel(self.model_name)
            
    def extract_text(self, image_path: str) -> str:
        if not self.model:
            return ""
            
        print(f"[GeminiOCR] Extracting text from {image_path} using {self.model_name}")
        try:
            img = Image.open(image_path)
            prompt = (
                "Extract the Classical Chinese (Sino-Nom) text from this document image. "
                "The text is written in vertical columns. You must read the columns from RIGHT to LEFT. "
                "Within each column, read characters from TOP to BOTTOM. "
                "Output ONLY the extracted text. Separate each column with a newline. "
                "Do not include any markdown formatting, explanations, or translations."
            )
            response = self.model.generate_content([prompt, img])
            if response and response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            print(f"[GeminiOCR] Error with {self.model_name}: {e}")
            if "404" in str(e) or "not found" in str(e):
                print("[GeminiOCR] Attempting fallback to 'gemini-pro-vision'...")
                try:
                    self.model_name = 'gemini-pro-vision'
                    self.model = genai.GenerativeModel(self.model_name)
                    response = self.model.generate_content([prompt, img])
                    if response and response.text:
                        return response.text.strip()
                except Exception as fallback_e:
                    print(f"[GeminiOCR] Fallback also failed: {fallback_e}")
                    
                    # Print available models for debugging
                    print("[GeminiOCR] Available models:")
                    try:
                        for m in genai.list_models():
                            if 'generateContent' in m.supported_generation_methods:
                                print(m.name)
                    except:
                        pass
            return ""

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
    def __init__(self, segment_sentences: bool = False):
        self.segment_sentences = segment_sentences
        try:
            # Monkey-patch to fix PaddlePaddle 3.0 compatibility with Paddlex
            import paddle
            try:
                if not hasattr(paddle.base.libpaddle.AnalysisConfig, 'set_optimization_level'):
                    paddle.base.libpaddle.AnalysisConfig.set_optimization_level = lambda *args, **kwargs: None
            except Exception:
                pass
                
            from paddleocr import PaddleOCR
            import paddleocr
            # Check version to pass correct arguments (PaddleOCR 3.7+ uses different kwarg names)
            version = getattr(paddleocr, '__version__', '2.8.1')
            
            if version.startswith('3'):
                # build_ocr_engine tắt tường minh doc_orientation_classify và
                # doc_unwarping — hai thứ này mặc định BẬT ở 3.x và làm hỏng ảnh cột.
                from ocr.ocr_pipeline import build_ocr_engine
                self.ocr = build_ocr_engine()
            else:
                self.ocr = PaddleOCR(
                    use_angle_cls=True, 
                    lang='chinese_cht',
                    det_db_unclip_ratio=1.8,
                    det_db_box_thresh=0.4,
                    drop_score=0.3
                )
        except ImportError:
            print("[Warning] PaddleOCR is not installed.")
            self.ocr = None

    def extract_text(self, image_path: str) -> str:
        if not self.ocr:
            return ""
        
        try:
            import os
            import sys
            
            from ocr.ocr_pipeline import ocr_sinonom_page
            
            # Call the advanced pipeline
            raw_output = ocr_sinonom_page(
                image_path=image_path,
                raw_columns=not self.segment_sentences,
                debug_ocr=False,
                output_dir=None,
                ocr_engine=self.ocr
            )
            
            if not self.segment_sentences:
                # Return raw text separated by newlines (OCR Only)
                # raw_output is a list of JSON dicts
                text_lines = [item['text'] for item in raw_output if isinstance(item, dict) and 'text' in item]
                return "\n".join(text_lines)
            else:
                # Return segmented sentences separated by newlines
                return "\n".join(raw_output)

        except Exception as e:
            import traceback
            print(f"[PaddleOCR] Advanced Pipeline Error: {e}")
            traceback.print_exc()
            return "Tự Hán (Paddle - Error)"


