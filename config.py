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


ENSEMBLE_CONFIG = {
    "scorers": {
        "labse": {
            "enabled": True,
            "weight": 0.40,
            "model_name": "sentence-transformers/LaBSE",
        },
        "vecalign": {
            "enabled": True,
            "weight": 0.30,
            "window_size": 3,
        },
        "bertalign": {
            "enabled": True,
            "weight": 0.15,
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
        "simalign": {
            "enabled": True,
            "weight": 0.15,
            "model": "xlmr",
            "top_k": 5,
        },
    },
    "dp": {
        "threshold": 0.32,
        "skip_penalty": 0.05,
        "max_merge_han": 15,
        "max_merge_viet": 2,
    },
    "qwen_verifier": {
        "enabled": True,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": True,
        "device_map": "auto",
        "uncertain_low": 0.32,
        "uncertain_high": 0.50,
        "keep_threshold": 3,
        "batch_size": 8,
    },
}


