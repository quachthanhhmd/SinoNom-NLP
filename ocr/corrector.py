import google.generativeai as genai
from core.interfaces import OCRCorrector

class GeminiOCRCorrector(OCRCorrector):
    def __init__(self, api_key: str):
        self.api_key = api_key
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-flash-latest')
        else:
            self.model = None
        
    def correct(self, text: str) -> str:
        if not self.model or not text.strip():
            return text
            
        print(f"[GeminiOCRCorrector] Correcting text of length {len(text)}")
        try:
            prompt = f"You are an expert in Classical Chinese (Sino-Nom). Please correct any obvious OCR errors in the following text. Only output the corrected text, nothing else. If there are no errors, output the text exactly as provided.\n\nText:\n{text}"
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[GeminiOCRCorrector] Error: {e}")
            return text
