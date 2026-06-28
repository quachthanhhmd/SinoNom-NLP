from typing import List, Dict
import numpy as np
import google.generativeai as genai
from core.interfaces import Aligner

class BERTAlignerWrapper(Aligner):
    """Wrapper for BERTAlign."""
    def __init__(self):
        pass

    def align(self, han_sentences: List[str], viet_sentences: List[str]) -> List[Dict[str, str]]:
        print(f"[BERTAlign] Mock Aligning {len(han_sentences)} Hán with {len(viet_sentences)} Việt.")
        aligned = []
        max_len = max(len(han_sentences), len(viet_sentences))
        for i in range(max_len):
            han = han_sentences[i] if i < len(han_sentences) else ""
            viet = viet_sentences[i] if i < len(viet_sentences) else ""
            aligned.append({
                "pair_id": f"pair_{i+1:04d}",
                "han_sentence": han,
                "viet_sentence": viet
            })
        return aligned

class TranslationCosineAligner(Aligner):
    """
    Advanced Aligner:
    1. Translates Hán to Việt using Gemini.
    2. Computes TF-IDF Cosine Similarity between translated Hán and original Việt.
    3. Uses Dynamic Time Warping (DTW) / Monotonic DP to find optimal alignment (handles m-n).
    """
    def __init__(self, api_key: str):
        if api_key and api_key != "your_api_key_here":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-flash-latest')
        else:
            self.model = None

    def translate_han_to_viet(self, han_sentences: List[str]) -> List[str]:
        if not self.model or not han_sentences:
            return han_sentences
            
        print("[Aligner] Translating Hán sentences to Việt for semantic matching...")
        translated = []
        # Process in batches to avoid enormous prompts, but for simplicity here we do it all
        try:
            prompt = "Translate the following Classical Chinese/Sino-Nom sentences into modern Vietnamese. Provide ONLY the translations, one per line, matching the exact number of input lines.\n\n"
            prompt += "\n".join(han_sentences)
            response = self.model.generate_content(prompt)
            lines = response.text.strip().split('\n')
            
            # If the model didn't return exact number, we pad or truncate
            if len(lines) < len(han_sentences):
                lines += [""] * (len(han_sentences) - len(lines))
            elif len(lines) > len(han_sentences):
                lines = lines[:len(han_sentences)]
                
            return [l.strip() for l in lines]
        except Exception as e:
            print(f"[Aligner] Translation error: {e}")
            return han_sentences

    def align(self, han_sentences: List[str], viet_sentences: List[str]) -> List[Dict[str, str]]:
        if not han_sentences and not viet_sentences:
            return []
            
        # 1. Translate
        translated_han = self.translate_han_to_viet(han_sentences)
        
        # 2. Vectorize
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            print("[Warning] scikit-learn not installed. Falling back to simple alignment.")
            return BERTAlignerWrapper().align(han_sentences, viet_sentences)
            
        print(f"[Aligner] Computing Cosine Similarity and aligning {len(han_sentences)} Hán with {len(viet_sentences)} Việt...")
        vectorizer = TfidfVectorizer().fit(translated_han + viet_sentences)
        han_vecs = vectorizer.transform(translated_han)
        viet_vecs = vectorizer.transform(viet_sentences)
        
        # similarity matrix (M x N)
        sim_matrix = cosine_similarity(han_vecs, viet_vecs)
        
        # 3. Dynamic Programming Alignment (Needleman-Wunsch variant for sentences)
        M, N = len(han_sentences), len(viet_sentences)
        
        # Cost matrix
        cost = np.zeros((M + 1, N + 1))
        # Base cases (insertion/deletion penalties)
        for i in range(1, M + 1):
            cost[i, 0] = i * 0.5
        for j in range(1, N + 1):
            cost[0, j] = j * 0.5
            
        # Pointers for backtracking
        ptr = np.zeros((M + 1, N + 1), dtype=int) # 1: diag, 2: up, 3: left
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                match_cost = cost[i-1, j-1] + (1.0 - sim_matrix[i-1, j-1])
                del_cost = cost[i-1, j] + 0.5
                ins_cost = cost[i, j-1] + 0.5
                
                min_c = min(match_cost, del_cost, ins_cost)
                cost[i, j] = min_c
                if min_c == match_cost:
                    ptr[i, j] = 1
                elif min_c == del_cost:
                    ptr[i, j] = 2
                else:
                    ptr[i, j] = 3
                    
        # Backtrack
        aligned_pairs = []
        i, j = M, N
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ptr[i, j] == 1:
                aligned_pairs.append((i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or ptr[i, j] == 2):
                aligned_pairs.append((i-1, -1)) # Hán only
                i -= 1
            else:
                aligned_pairs.append((-1, j-1)) # Viet only
                j -= 1
                
        aligned_pairs.reverse()
        
        # 4. Format Output
        results = []
        idx = 1
        for h_idx, v_idx in aligned_pairs:
            han = han_sentences[h_idx] if h_idx != -1 else ""
            viet = viet_sentences[v_idx] if v_idx != -1 else ""
            results.append({
                "pair_id": f"pair_{idx:04d}",
                "han_sentence": han,
                "viet_sentence": viet
            })
            idx += 1
            
        return results
