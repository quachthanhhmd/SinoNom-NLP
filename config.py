import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Directories
    HAN_DIR = os.environ.get("HAN_DIR", "dataset/china")
    VIET_DIR = os.environ.get("VIET_DIR", "dataset/vietnam")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
    
    # API Keys
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    KANDIAN_TOKEN = os.environ.get("KANDIAN_TOKEN", "")
    KANDIAN_EMAIL = os.environ.get("KANDIAN_EMAIL", "")
    GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
    
    # Flags
    ENABLE_OCR = True
    ENABLE_CORRECTION = True
