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
        try:
            prompt = "Translate the following Classical Chinese/Sino-Nom sentences into modern Vietnamese. Provide ONLY the translations, one per line, matching the exact number of input lines.\n\n"
            prompt += "\n".join(han_sentences)
            response = self.model.generate_content(prompt)
            lines = response.text.strip().split('\n')
            
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
            
        translated_han = self.translate_han_to_viet(han_sentences)
        
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
        
        sim_matrix = cosine_similarity(han_vecs, viet_vecs)
        
        M, N = len(han_sentences), len(viet_sentences)
        
        cost = np.zeros((M + 1, N + 1))
        for i in range(1, M + 1):
            cost[i, 0] = i * 0.5
        for j in range(1, N + 1):
            cost[0, j] = j * 0.5
            
        ptr = np.zeros((M + 1, N + 1), dtype=int)
        
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
                    
        aligned_pairs = []
        i, j = M, N
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ptr[i, j] == 1:
                aligned_pairs.append((i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or ptr[i, j] == 2):
                aligned_pairs.append((i-1, -1))
                i -= 1
            else:
                aligned_pairs.append((-1, j-1))
                j -= 1
                
        aligned_pairs.reverse()
        
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

class EmbeddingSentenceAligner(Aligner):
    """
    Sentence-level Aligner using:
    1. Multilingual Sentence Embeddings (LaBSE) to compute Cosine Similarity between Han and Viet directly.
    2. Generalized Dynamic Programming supporting W-1 and 1-L mapping for flexible sentence merging.
    3. Performance-optimized Precomputation (vector aggregation & normalization pre-calculated outside DP loop).
    """
    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str = None):
        self.model_name = model_name
        self.device = device
        self._embed_model = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            print(f"[Aligner] Loading embedding model: {self.model_name}...")
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                print("[Error] sentence-transformers is not installed.")
                raise
        return self._embed_model

    def align(self, han_sentences: List[str], viet_sentences: List[str]) -> List[Dict[str, str]]:
        if not han_sentences or not viet_sentences:
            return []

        M, N = len(han_sentences), len(viet_sentences)
        
        # Encode sentences using LaBSE directly
        print(f"[Aligner] Encoding {M} Han and {N} Viet sentences using {self.model_name}...")
        han_embeds = self.embed_model.encode(han_sentences, convert_to_numpy=True, show_progress_bar=False)
        viet_embeds = self.embed_model.encode(viet_sentences, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalize baseline embeddings
        han_embeds_norm = han_embeds / np.linalg.norm(han_embeds, axis=1, keepdims=True)
        viet_embeds_norm = viet_embeds / np.linalg.norm(viet_embeds, axis=1, keepdims=True)
        
        # DP Hyperparameters
        threshold = 0.38 # lowered threshold to capture long-short alignments securely after preface filtering
        skip_penalty = 0.05 # small penalty to avoid excessive skipping
        
        # Max merge configurations
        max_merge_han = 15 # supports merging up to 15 Han sentences for 1 long Viet sentence
        max_merge_viet = 2 # supports merging up to 2 Viet sentences for 1 Han sentence

        print(f"[Aligner] Precomputing aggregated embeddings (max_merge_han={max_merge_han}, max_merge_viet={max_merge_viet})...")
        
        # 1. Precompute and normalize all merged Han embeddings
        # han_merged_norms[k][i] will store the normalized embedding for merging k sentences ending at index i.
        # i ranges from 0 to M-1. k ranges from 1 to max_merge_han.
        # If the range [i-k+1, i] is out of bounds (i-k+1 < 0), we store a zero vector.
        han_merged_norms = {}
        for k in range(1, max_merge_han + 1):
            merged_k = np.zeros((M, 768))
            for i in range(M):
                start = i - k + 1
                if start >= 0:
                    agg = np.sum(han_embeds[start:i+1], axis=0)
                    norm = np.linalg.norm(agg)
                    if norm > 0:
                        merged_k[i] = agg / norm
            han_merged_norms[k] = merged_k

        # 2. Precompute and normalize all merged Viet embeddings
        # viet_merged_norms[l][j] will store the normalized embedding for merging l sentences ending at index j.
        # j ranges from 0 to N-1. l ranges from 1 to max_merge_viet.
        viet_merged_norms = {}
        for l in range(1, max_merge_viet + 1):
            merged_l = np.zeros((N, 768))
            for j in range(N):
                start = j - l + 1
                if start >= 0:
                    agg = np.sum(viet_embeds[start:j+1], axis=0)
                    norm = np.linalg.norm(agg)
                    if norm > 0:
                        merged_l[j] = agg / norm
            viet_merged_norms[l] = merged_l
        
        # Helper functions to compute similarities using precomputed normalized vectors
        def get_sim_k_1(i_start, i_end, j):
            # i_start to i_end corresponds to merging k sentences ending at i_end.
            k = i_end - i_start + 1
            return np.dot(han_merged_norms[k][i_end], viet_embeds_norm[j])
            
        def get_sim_1_l(i, j_start, j_end):
            # j_start to j_end corresponds to merging l sentences ending at j_end.
            l = j_end - j_start + 1
            return np.dot(han_embeds_norm[i], viet_merged_norms[l][j_end])

        # Match score helper functions
        def get_score_k_1(i_start, i_end, j):
            sim = get_sim_k_1(i_start, i_end, j)
            return sim - threshold if sim >= threshold else -10.0
            
        def get_score_1_l(i, j_start, j_end):
            sim = get_sim_1_l(i, j_start, j_end)
            return sim - threshold if sim >= threshold else -10.0

        # DP matrix
        dp = np.full((M + 1, N + 1), -1e9)
        # ptr will store a tuple: (move_type, k, l)
        # move_type: 1 for k-1 mapping, 2 for 1-l mapping, 4 for Skip Han, 5 for Skip Viet
        ptr = np.zeros((M + 1, N + 1), dtype=object)
        dp[0][0] = 0.0

        print("[Aligner] Running Dynamic Programming alignment...")

        # Fill DP table
        for i in range(M + 1):
            for j in range(N + 1):
                if i == 0 and j == 0:
                    continue
                
                # Option 4: Skip Han (H[i-1] mapped to empty)
                if i > 0:
                    val = dp[i-1][j] - skip_penalty
                    if val > dp[i][j]:
                        dp[i][j] = val
                        ptr[i][j] = (4, 1, 0)
                        
                # Option 5: Skip Viet (V[j-1] mapped to empty)
                if j > 0:
                    val = dp[i][j-1] - skip_penalty
                    if val > dp[i][j]:
                        dp[i][j] = val
                        ptr[i][j] = (5, 0, 1)
                        
                # Option 1: k-1 mapping (merge k Han sentences to 1 Viet sentence)
                for k in range(1, max_merge_han + 1):
                    if i >= k and j >= 1:
                        score = get_score_k_1(i - k, i - 1, j - 1)
                        val = dp[i - k][j - 1] + score
                        if val > dp[i][j]:
                            dp[i][j] = val
                            ptr[i][j] = (1, k, 1)
                            
                # Option 2: 1-l mapping (merge l Viet sentences to 1 Han sentence)
                for l in range(2, max_merge_viet + 1):
                    if i >= 1 and j >= l:
                        score = get_score_1_l(i - 1, j - l, j - 1)
                        val = dp[i - 1][j - l] + score
                        if val > dp[i][j]:
                            dp[i][j] = val
                            ptr[i][j] = (2, 1, l)

        # Backtracking
        i, j = M, N
        aligned_pairs = []
        
        while i > 0 or j > 0:
            move = ptr[i][j]
            if not move:
                if i > 0:
                    aligned_pairs.append(([i-1], [], 0.0))
                    i -= 1
                else:
                    aligned_pairs.append(([], [j-1], 0.0))
                    j -= 1
                continue
                
            move_type, k, l = move
            if move_type == 1: # k-1 mapping
                sim = get_sim_k_1(i - k, i - 1, j - 1)
                aligned_pairs.append((list(range(i - k, i)), [j - 1], sim))
                i -= k
                j -= 1
            elif move_type == 2: # 1-l mapping
                sim = get_sim_1_l(i - 1, j - l, j - 1)
                aligned_pairs.append(([i - 1], list(range(j - l, j)), sim))
                i -= 1
                j -= l
            elif move_type == 4: # Skip Han
                aligned_pairs.append(([i - 1], [], 0.0))
                i -= 1
            elif move_type == 5: # Skip Viet
                aligned_pairs.append(([], [j - 1], 0.0))
                j -= 1
            else:
                if i > 0:
                    aligned_pairs.append(([i-1], [], 0.0))
                    i -= 1
                else:
                    aligned_pairs.append(([], [j-1], 0.0))
                    j -= 1
                    
        aligned_pairs.reverse()
        
        # Format output
        results = []
        idx = 1
        for h_idxs, v_idxs, sim in aligned_pairs:
            han_txt = " ".join([han_sentences[h] for h in h_idxs]) if h_idxs else ""
            viet_txt = " ".join([viet_sentences[v] for v in v_idxs]) if v_idxs else ""
            
            if not han_txt and not viet_txt:
                continue
                
            results.append({
                "pair_id": f"pair_{idx:06d}",
                "han_sentence": han_txt,
                "viet_sentence": viet_txt,
                "similarity_score": round(float(sim), 4)
            })
            idx += 1
            
        return results
