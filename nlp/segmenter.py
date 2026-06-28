import re
from typing import List
from core.interfaces import Segmenter

class RegexSegmenter(Segmenter):
    """A basic segmenter using regex for common punctuation."""
    
    def __init__(self, lang="han"):
        self.lang = lang
        if self.lang == "han":
            # Punctuation for Classical Chinese/Sino-Nom, including newlines
            self.pattern = re.compile(r'([。！？\.\!\?\n])')
        else:
            # Punctuation for Vietnamese, including newlines
            self.pattern = re.compile(r'([.!?\n])\s*')
            
    def segment(self, text: str) -> List[str]:
        # Simple split by punctuation, keeping the punctuation
        parts = self.pattern.split(text)
        sentences = []
        current = ""
        for part in parts:
            if self.pattern.match(part):
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
            # Fallback to simple regex if underthesea is missing
            return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            
        print("[Underthesea] Segmenting Vietnamese text...")
        # Underthesea sent_tokenize returns a list of sentences
        # We replace newlines with space first to prevent them from breaking sentences unnaturally,
        # or we can keep them. Usually for PDF extraction, newlines are arbitrary.
        clean_text = text.replace("\n", " ")
        sentences = self.tokenizer(clean_text)
        return [s.strip() for s in sentences if s.strip()]
