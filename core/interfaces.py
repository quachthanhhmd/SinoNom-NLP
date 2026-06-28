from abc import ABC, abstractmethod
from typing import List, Dict

class OCRProvider(ABC):
    """Abstract base class for OCR extraction."""
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        pass

class OCRCorrector(ABC):
    """Abstract base class for OCR correction."""
    @abstractmethod
    def correct(self, text: str) -> str:
        pass

class Segmenter(ABC):
    """Abstract base class for sentence segmentation."""
    @abstractmethod
    def segment(self, text: str) -> List[str]:
        pass

class Aligner(ABC):
    """Abstract base class for Hán-Việt alignment."""
    @abstractmethod
    def align(self, han_sentences: List[str], viet_sentences: List[str]) -> List[Dict[str, str]]:
        """
        Returns a list of dicts:
        [{'pair_id': '...', 'han_sentence': '...', 'viet_sentence': '...'}]
        """
        pass
