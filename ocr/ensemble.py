from typing import List
from core.interfaces import OCRProvider

class EnsembleOCR(OCRProvider):
    def __init__(self, providers: List[OCRProvider]):
        self.providers = providers

    def extract_text(self, image_path: str) -> str:
        results = []
        for provider in self.providers:
            try:
                text = provider.extract_text(image_path)
                results.append(text)
            except Exception as e:
                print(f"Error with provider {type(provider).__name__}: {e}")
        
        return self._vote(results)

    def _vote(self, results: List[str]) -> str:
        """
        Improved voting logic:
        1. Filter out mock/error messages.
        2. Pick the result with the highest count of CJK (Sino-Nom) characters.
        """
        if not results:
            return ""
            
        # 1. Filter out known mock/error responses
        valid_results = [r for r in results if not r.startswith("Tự Hán (")]
        
        # If all were errors/mocks, fallback to original list
        if not valid_results:
            valid_results = results
            
        # 2. Score by CJK (Sino-Nom) character count instead of raw length
        import re
        def cjk_count(text: str) -> int:
            # Matches standard CJK characters and Extension A
            # \u4e00-\u9fff : CJK Unified Ideographs
            # \u3400-\u4dbf : CJK Unified Ideographs Extension A
            return len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
            
        return max(valid_results, key=cjk_count)
