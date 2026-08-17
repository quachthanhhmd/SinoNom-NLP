import re
from typing import List
from core.interfaces import Segmenter

class RegexSegmenter(Segmenter):
    """A basic segmenter using regex for common punctuation."""
    
    def __init__(self, lang="han"):
        self.lang = lang
        if self.lang == "han":
            # OCR line wraps are soft layout hints, not sentence boundaries.
            self.pattern = re.compile(r'([。！？\.\!\?])')
        else:
            self.pattern = re.compile(r'([.!?])\s*')
            
    def segment(self, text: str) -> List[str]:
        sentences = []
        # Preserve paragraph boundaries while joining arbitrary OCR/PDF line
        # wraps inside a paragraph.
        for paragraph in re.split(r'\n\s*\n+', text):
            paragraph = re.sub(r'\s*\n\s*', ' ', paragraph).strip()
            if not paragraph:
                continue
            parts = self.pattern.split(paragraph)
            current = ""
            for part in parts:
                if self.pattern.fullmatch(part):
                    sentences.append(current + part)
                    current = ""
                else:
                    current += part
            if current.strip():
                sentences.append(current)
            
        return [s.strip() for s in sentences if s.strip()]

class UndertheseaSegmenter(Segmenter):
    """Segmenter that uses Underthesea for Vietnamese text."""
    
    def __init__(self):
        try:
            import underthesea
            self.tokenizer = underthesea.sent_tokenize
        except ImportError:
            print("[Warning] underthesea is not installed. Falling back to simple split.")
            self.tokenizer = None
            
    def segment(self, text: str) -> List[str]:
        if not self.tokenizer:
            return RegexSegmenter(lang="viet").segment(text)
            
        print("[Underthesea] Segmenting Vietnamese text...")
        sentences = []
        for paragraph in re.split(r'\n\s*\n+', text):
            clean_text = re.sub(r'\s*\n\s*', ' ', paragraph).strip()
            if clean_text:
                sentences.extend(self.tokenizer(clean_text))
        return [s.strip() for s in sentences if s.strip()]
