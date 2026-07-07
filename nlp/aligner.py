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

class EmbeddingSentenceAligner(Aligner):
    """
    Sentence-level Aligner using:
    1. Helsinki-NLP/opus-mt-zh-vi translation model to translate Han sentences to Viet.
    2. Multilingual Sentence Embeddings (LaBSE) to compute Cosine Similarity between Viet translation and Viet target.
    3. Dynamic Programming for optimal m-n alignment, outputting similarity_score.
    """
    def __init__(self, model_name: str = "sentence-transformers/LaBSE", translation_model: str = "Helsinki-NLP/opus-mt-zh-vi", device: str = None):
        self.model_name = model_name
        self.translation_model = translation_model
        self.device = device
        self._embed_model = None
        self._trans_tokenizer = None
        self._trans_model = None

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

    def load_translation_model(self):
        if self._trans_model is None:
            print(f"[Aligner] Loading translation model: {self.translation_model}...")
            try:
                from transformers import MarianMTModel, MarianTokenizer
                import torch
                # Determine device
                dev = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
                self._trans_tokenizer = MarianTokenizer.from_pretrained(self.translation_model)
                self._trans_model = MarianMTModel.from_pretrained(self.translation_model).to(dev)
            except Exception as e:
                print(f"[Warning] Failed to load translation model offline: {e}. Falling back to direct embedding alignment.")
                self._trans_model = False # Marker for failed load

    def translate_han_to_viet(self, han_sentences: List[str]) -> List[str]:
        self.load_translation_model()
        if not self._trans_model: # False or None due to load failure
            return han_sentences

        import torch
        dev = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        total = len(han_sentences)
        print(f"[Aligner] Translating {total} Han sentences to Viet...")
        translated = []
        batch_size = 64
        
        for i in range(0, total, batch_size):
            batch = han_sentences[i:i+batch_size]
            inputs = self._trans_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(dev)
            with torch.no_grad():
                translated_tokens = self._trans_model.generate(
                    **inputs,
                    num_beams=1,
                    max_new_tokens=100,
                    early_stopping=True
                )
            decoded = self._trans_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated.extend([d.strip() for d in decoded])
            
            processed = min(i + batch_size, total)
            print(f"  -> Translated {processed}/{total} sentences ({(processed / total) * 100:.1f}%)")
            
        return translated

    def align(self, han_sentences: List[str], viet_sentences: List[str]) -> List[Dict[str, str]]:
        if not han_sentences or not viet_sentences:
            return []

        M, N = len(han_sentences), len(viet_sentences)
        
        # Step 1: Translate Han sentences to Viet
        translated_han = self.translate_han_to_viet(han_sentences)
        
        # Step 2: Encode sentences using LaBSE
        print(f"[Aligner] Encoding translated Han and target Viet sentences using {self.model_name}...")
        han_embeds = self.embed_model.encode(translated_han, convert_to_numpy=True, show_progress_bar=False)
        viet_embeds = self.embed_model.encode(viet_sentences, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalize embeddings
        han_embeds_norm = han_embeds / np.linalg.norm(han_embeds, axis=1, keepdims=True)
        viet_embeds_norm = viet_embeds / np.linalg.norm(viet_embeds, axis=1, keepdims=True)
        
        # DP matrix
        dp = np.full((M + 1, N + 1), -1e9)
        ptr = np.zeros((M + 1, N + 1), dtype=int)
        dp[0][0] = 0.0
        
        # DP Hyperparameters
        threshold = 0.42
        skip_penalty = 0.05 # small penalty to avoid excessive skipping
        
        # Helper to compute normalized cosine similarity of aggregated vectors
        def get_sim_1_1(i, j):
            return np.dot(han_embeds_norm[i], viet_embeds_norm[j])
            
        def get_sim_1_2(i, j_start, j_end):
            v_agg = np.sum(viet_embeds[j_start:j_end+1], axis=0)
            norm = np.linalg.norm(v_agg)
            if norm == 0:
                return 0.0
            v_agg_norm = v_agg / norm
            return np.dot(han_embeds_norm[i], v_agg_norm)
            
        def get_sim_2_1(i_start, i_end, j):
            h_agg = np.sum(han_embeds[i_start:i_end+1], axis=0)
            norm = np.linalg.norm(h_agg)
            if norm == 0:
                return 0.0
            h_agg_norm = h_agg / norm
            return np.dot(h_agg_norm, viet_embeds_norm[j])

        # Match score helpers (returns penalty for mismatches)
        def get_score_1_1(i, j):
            sim = get_sim_1_1(i, j)
            return sim - threshold if sim >= threshold else -1.0
            
        def get_score_1_2(i, j_start, j_end):
            sim = get_sim_1_2(i, j_start, j_end)
            return sim - threshold if sim >= threshold else -1.0
            
        def get_score_2_1(i_start, i_end, j):
            sim = get_sim_2_1(i_start, i_end, j)
            return sim - threshold if sim >= threshold else -1.0

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
                        ptr[i][j] = 4
                        
                # Option 5: Skip Viet (V[j-1] mapped to empty)
                if j > 0:
                    val = dp[i][j-1] - skip_penalty
                    if val > dp[i][j]:
                        dp[i][j] = val
                        ptr[i][j] = 5
                        
                # Option 1: 1-1 mapping
                if i > 0 and j > 0:
                    val = dp[i-1][j-1] + get_score_1_1(i-1, j-1)
                    if val > dp[i][j]:
                        dp[i][j] = val
                        ptr[i][j] = 1
                        
                # Option 2: 1-2 mapping (1 Han sentence mapped to 2 Viet sentences)
                if i > 0 and j > 1:
                    val = dp[i-1][j-2] + get_score_1_2(i-1, j-2, j-1)
                    if val > dp[i][j]:
                        dp[i][j] = val
                        ptr[i][j] = 2
                        
                # Option 3: 2-1 mapping (2 Han sentences mapped to 1 Viet sentence)
                if i > 1 and j > 0:
                    val = dp[i-2][j-1] + get_score_2_1(i-2, i-1, j-1)
                    if val > dp[i][j]:
                        dp[i][j] = val
                        ptr[i][j] = 3

        # Backtracking
        i, j = M, N
        aligned_pairs = []
        
        while i > 0 or j > 0:
            move = ptr[i][j]
            if move == 1: # 1-1
                sim = get_sim_1_1(i-1, j-1)
                aligned_pairs.append(([i-1], [j-1], sim))
                i -= 1
                j -= 1
            elif move == 2: # 1-2
                sim = get_sim_1_2(i-1, j-2, j-1)
                aligned_pairs.append(([i-1], [j-2, j-1], sim))
                i -= 1
                j -= 2
            elif move == 3: # 2-1
                sim = get_sim_2_1(i-2, i-1, j-1)
                aligned_pairs.append(([i-2, i-1], [j-1], sim))
                i -= 2
                j -= 1
            elif move == 4: # Skip Han
                aligned_pairs.append(([i-1], [], 0.0))
                i -= 1
            elif move == 5: # Skip Viet
                aligned_pairs.append(([], [j-1], 0.0))
                j -= 1
            else:
                if i > 0 and j > 0:
                    sim = get_sim_1_1(i-1, j-1)
                    aligned_pairs.append(([i-1], [j-1], sim))
                    i -= 1
                    j -= 1
                elif i > 0:
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

