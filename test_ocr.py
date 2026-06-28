import os
from config import Config
from ocr.providers import GoogleVisionOCR, PaddleOCRProvider, KanDianGuJiOCR
from ocr.ensemble import EnsembleOCR

def test_ocr(image_path):
    print(f"--- Testing OCR on {image_path} ---")
    
    kandian_token = getattr(Config, 'KANDIAN_TOKEN', '')
    kandian_email = getattr(Config, 'KANDIAN_EMAIL', '')
    providers = []
    
    # print("\n1. Initializing KanDianGuJiOCR...")
    # kandian = KanDianGuJiOCR(token=kandian_token, email=kandian_email)
    # providers.append(kandian)
    # print("KanDianGuJiOCR Result:", repr(kandian.extract_text(image_path)))
    
    print("\n2. Initializing PaddleOCR...")
    paddle = PaddleOCRProvider()
    providers.append(paddle)
    print("PaddleOCR Result:", repr(paddle.extract_text(image_path)))
    
    # google_vision = GoogleVisionOCR(api_key=Config.GOOGLE_VISION_API_KEY)
    # providers.append(google_vision)
    # print("GoogleVisionOCR Result:", repr(google_vision.extract_text(image_path)))
        
    print("\n4. Testing Ensemble OCR (Chon ket qua tot nhat)...")
    ensemble = EnsembleOCR(providers=providers)
    
    result = ensemble.extract_text(image_path)
    
    print("\n========== FINAL EXTRACTED TEXT ==========")
    print(result)
    print("==========================================")

if __name__ == "__main__":
    target_img = "dataset/china/q1/page_001.jpg"
    if os.path.exists(target_img):
        test_ocr(target_img)
    else:
        print(f"Error: {target_img} not found.")
