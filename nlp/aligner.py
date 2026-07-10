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
        
        # Use dynamic embedding dimension to support any model (not just LaBSE 768-dim)
        D = han_embeds.shape[1]
        
        # Vectorized precomputation using cumulative sums (np.cumsum).
        # padded_cumsum[i+1] = sum of embeddings[0..i], padded_cumsum[0] = zero vector.
        # So sum of embeddings from i_start to i_end = padded_cumsum[i_end+1] - padded_cumsum[i_start]
        padded_han_cumsum = np.vstack([np.zeros((1, D)), np.cumsum(han_embeds, axis=0)])  # shape (M+1, D)
        padded_viet_cumsum = np.vstack([np.zeros((1, D)), np.cumsum(viet_embeds, axis=0)])  # shape (N+1, D)
        
        # 1. Precompute and normalize all merged Han embeddings
        # han_merged_norms[k][i] stores the normalized merged embedding for k sentences ending at index i.
        # Valid only when i >= k-1 (i.e., there are at least k sentences before and including i).
        han_merged_norms = {}
        for k in range(1, max_merge_han + 1):
            merged_k = np.zeros((M, D))
            if k <= M:
                # All windows of size k: agg[r] = sum of han_embeds[r-k+1 .. r] for r in [k-1, M-1]
                agg = padded_han_cumsum[k:M+1] - padded_han_cumsum[0:M-k+1]  # shape (M-k+1, D)
                norms = np.linalg.norm(agg, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero
                merged_k[k-1:M] = agg / norms  # store at valid index positions [k-1, M-1]
            han_merged_norms[k] = merged_k

        # 2. Precompute and normalize all merged Viet embeddings
        # viet_merged_norms[l][j] stores the normalized merged embedding for l sentences ending at index j.
        viet_merged_norms = {}
        for l in range(1, max_merge_viet + 1):
            merged_l = np.zeros((N, D))
            if l <= N:
                agg = padded_viet_cumsum[l:N+1] - padded_viet_cumsum[0:N-l+1]  # shape (N-l+1, D)
                norms = np.linalg.norm(agg, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                merged_l[l-1:N] = agg / norms
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


class EnsembleSentenceAligner(Aligner):
    """
    Sentence-level Aligner using an Ensemble of:
    1. LaBSE (bi-encoder similarity)
    2. Vecalign (context-window overlap embeddings)
    3. BERTAlign (paraphrase-multilingual bi-encoder similarity)
    4. SimAlign (word-level contextual alignment similarity)

    Fuses similarity matrices using EnsembleFuser and runs monotonic DP.
    Phase 2 LLM verification (Qwen) is executed as a separate step.
    """
    def __init__(self, device: str = None):
        self.device = device
        # Load config lazily inside align
        self.config = None

    def align(self, han_sentences: List[str], viet_sentences: List[str]) -> List[Dict[str, str]]:
        if not han_sentences or not viet_sentences:
            return []

        # Lazy load config
        from config import ENSEMBLE_CONFIG
        self.config = ENSEMBLE_CONFIG

        # Import scorers and fuser lazily
        from .scorers import (
            LaBSEScorer,
            VecalignScorer,
            BERTAlignScorer,
            SimAlignScorer,
            EnsembleFuser
        )

        M, N = len(han_sentences), len(viet_sentences)
        print(f"[Aligner] === Phase 1: Embedding Ensemble ({M} Han, {N} Viet) ===")
        t_phase1 = __import__('time').time()

        # Extract scorer configurations
        scorers_conf = self.config.get("scorers", {})
        dp_conf = self.config.get("dp", {})
        
        threshold = dp_conf.get("threshold", 0.38)
        skip_penalty = dp_conf.get("skip_penalty", 0.05)
        max_merge_han = dp_conf.get("max_merge_han", 15)
        max_merge_viet = dp_conf.get("max_merge_viet", 2)

        # ----------------------------------------------------
        # 1. Compute Base Embeddings & Similarity Matrices
        # ----------------------------------------------------
        score_matrices = {}
        active_weights = {}

        # 1.1 LaBSE & Vecalign share the same LaBSE model
        labse_conf = scorers_conf.get("labse", {})
        vecalign_conf = scorers_conf.get("vecalign", {})

        labse_han_norm, labse_viet_norm = None, None
        
        if labse_conf.get("enabled", True) or vecalign_conf.get("enabled", True):
            labse_model_name = labse_conf.get("model_name", "sentence-transformers/LaBSE")
            labse_scorer = LaBSEScorer(model_name=labse_model_name, device=self.device)
            
            # Encode once and reuse
            labse_han_norm = labse_scorer.encode(han_sentences, "Han (LaBSE)")
            labse_viet_norm = labse_scorer.encode(viet_sentences, "Viet (LaBSE)")

            if labse_conf.get("enabled", True):
                score_matrices["labse"] = labse_scorer.score(
                    han_sentences, viet_sentences, labse_han_norm, labse_viet_norm
                )
                active_weights["labse"] = labse_conf.get("weight", 0.20)

            if vecalign_conf.get("enabled", True):
                window_size = vecalign_conf.get("window_size", 3)
                vecalign_scorer = VecalignScorer(window_size=window_size)
                score_matrices["vecalign"] = vecalign_scorer.score(
                    han_sentences, viet_sentences, labse_han_norm, labse_viet_norm
                )
                active_weights["vecalign"] = vecalign_conf.get("weight", 0.30)

        # 1.2 BERTAlign (using paraphrase-multilingual)
        bert_conf = scorers_conf.get("bertalign", {})
        bert_han_norm, bert_viet_norm = None, None
        if bert_conf.get("enabled", True):
            bert_model_name = bert_conf.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            bert_scorer = BERTAlignScorer(model_name=bert_model_name, device=self.device)
            # We encode and save embeddings for merged precomputations later
            bert_han_norm = bert_scorer._encode_normalized(han_sentences, "Han (BERTAlign)")
            bert_viet_norm = bert_scorer._encode_normalized(viet_sentences, "Viet (BERTAlign)")
            
            # Use precomputed to avoid re-encoding
            score_matrices["bertalign"] = bert_han_norm @ bert_viet_norm.T
            active_weights["bertalign"] = bert_conf.get("weight", 0.25)

        # 1.3 SimAlign (Word-level contextual alignment)
        simalign_conf = scorers_conf.get("simalign", {})
        simalign_matrix = None
        if simalign_conf.get("enabled", True):
            simalign_scorer = SimAlignScorer(
                model_name=simalign_conf.get("model", "xlmr"),
                top_k=simalign_conf.get("top_k", 5)
            )
            if simalign_scorer.is_available():
                # We need a reference matrix to filter top_k candidate pairs.
                ref_matrix = score_matrices.get("labse", list(score_matrices.values())[0] if score_matrices else None)
                simalign_matrix = simalign_scorer.score(han_sentences, viet_sentences, ref_matrix)
                score_matrices["simalign"] = simalign_matrix
                active_weights["simalign"] = simalign_conf.get("weight", 0.25)
            else:
                print("[Aligner] Warning: simalign is not installed. SimAlign scorer disabled.")

        # Initialize fuser
        fuser = EnsembleFuser(active_weights)

        # ----------------------------------------------------
        # 2. Vectorized Group Similarity Precomputation
        # ----------------------------------------------------
        print(f"[Aligner] Precomputing group similarities (max_merge_han={max_merge_han}, max_merge_viet={max_merge_viet})...")
        t_pre = __import__('time').time()

        # Base cumsums for fast embedding sum/mean
        D_labse = labse_han_norm.shape[1] if labse_han_norm is not None else 0
        D_bert = bert_han_norm.shape[1] if bert_han_norm is not None else 0

        # Sum of normalized embeddings is standard and extremely close in direction.
        padded_han_labse = np.vstack([np.zeros((1, D_labse)), np.cumsum(labse_han_norm, axis=0)]) if labse_han_norm is not None else None
        padded_viet_labse = np.vstack([np.zeros((1, D_labse)), np.cumsum(labse_viet_norm, axis=0)]) if labse_viet_norm is not None else None

        padded_han_bert = np.vstack([np.zeros((1, D_bert)), np.cumsum(bert_han_norm, axis=0)]) if bert_han_norm is not None else None
        padded_viet_bert = np.vstack([np.zeros((1, D_bert)), np.cumsum(bert_viet_norm, axis=0)]) if bert_viet_norm is not None else None

        # Vecalign overlap embeddings
        vecalign_han_norm = vecalign_scorer._overlap_embeddings(labse_han_norm) if "vecalign" in score_matrices else None
        vecalign_viet_norm = vecalign_scorer._overlap_embeddings(labse_viet_norm) if "vecalign" in score_matrices else None
        padded_han_vecalign = np.vstack([np.zeros((1, D_labse)), np.cumsum(vecalign_han_norm, axis=0)]) if vecalign_han_norm is not None else None
        padded_viet_vecalign = np.vstack([np.zeros((1, D_labse)), np.cumsum(vecalign_viet_norm, axis=0)]) if vecalign_viet_norm is not None else None

        # SimAlign cumulative sums for range average
        simalign_cumsum_han = np.vstack([np.zeros((1, N)), np.cumsum(simalign_matrix, axis=0)]) if simalign_matrix is not None else None
        simalign_cumsum_viet = np.hstack([np.zeros((M, 1)), np.cumsum(simalign_matrix, axis=1)]) if simalign_matrix is not None else None

        # Precompute k-1 mappings (merging k Han sentences to 1 Viet sentence)
        fused_sim_k_1 = {}
        for k in range(1, max_merge_han + 1):
            k_scores = {}
            
            # LaBSE
            if "labse" in score_matrices:
                merged_labse_k = np.zeros((M, D_labse))
                if k <= M:
                    agg = padded_han_labse[k:M+1] - padded_han_labse[0:M-k+1]
                    norms = np.linalg.norm(agg, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    merged_labse_k[k-1:M] = agg / norms
                k_scores["labse"] = merged_labse_k @ labse_viet_norm.T

            # Vecalign
            if "vecalign" in score_matrices:
                merged_vecalign_k = np.zeros((M, D_labse))
                if k <= M:
                    agg = padded_han_vecalign[k:M+1] - padded_han_vecalign[0:M-k+1]
                    norms = np.linalg.norm(agg, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    merged_vecalign_k[k-1:M] = agg / norms
                k_scores["vecalign"] = merged_vecalign_k @ vecalign_viet_norm.T

            # BERTAlign
            if "bertalign" in score_matrices:
                merged_bert_k = np.zeros((M, D_bert))
                if k <= M:
                    agg = padded_han_bert[k:M+1] - padded_han_bert[0:M-k+1]
                    norms = np.linalg.norm(agg, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    merged_bert_k[k-1:M] = agg / norms
                k_scores["bertalign"] = merged_bert_k @ bert_viet_norm.T

            # SimAlign (Average of range)
            if "simalign" in score_matrices:
                sim_k_1_simalign = np.zeros((M, N))
                if k <= M:
                    agg = (simalign_cumsum_han[k:M+1] - simalign_cumsum_han[0:M-k+1]) / k
                    sim_k_1_simalign[k-1:M] = agg
                k_scores["simalign"] = sim_k_1_simalign

            # Fuse them
            fused_sim_k_1[k] = fuser.fuse(k_scores)

        # Precompute 1-l mappings (merging l Viet sentences to 1 Han sentence)
        fused_sim_1_l = {}
        for l in range(1, max_merge_viet + 1):
            l_scores = {}
            
            # LaBSE
            if "labse" in score_matrices:
                merged_labse_l = np.zeros((N, D_labse))
                if l <= N:
                    agg = padded_viet_labse[l:N+1] - padded_viet_labse[0:N-l+1]
                    norms = np.linalg.norm(agg, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    merged_labse_l[l-1:N] = agg / norms
                l_scores["labse"] = labse_han_norm @ merged_labse_l.T

            # Vecalign
            if "vecalign" in score_matrices:
                merged_vecalign_l = np.zeros((N, D_labse))
                if l <= N:
                    agg = padded_viet_vecalign[l:N+1] - padded_viet_vecalign[0:N-l+1]
                    norms = np.linalg.norm(agg, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    merged_vecalign_l[l-1:N] = agg / norms
                l_scores["vecalign"] = vecalign_han_norm @ merged_vecalign_l.T

            # BERTAlign
            if "bertalign" in score_matrices:
                merged_bert_l = np.zeros((N, D_bert))
                if l <= N:
                    agg = padded_viet_bert[l:N+1] - padded_viet_bert[0:N-l+1]
                    norms = np.linalg.norm(agg, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    merged_bert_l[l-1:N] = agg / norms
                l_scores["bertalign"] = bert_han_norm @ merged_bert_l.T

            # SimAlign (Average of range)
            if "simalign" in score_matrices:
                sim_1_l_simalign = np.zeros((M, N))
                if l <= N:
                    agg = (simalign_cumsum_viet[:, l:N+1] - simalign_cumsum_viet[:, 0:N-l+1]) / l
                    sim_1_l_simalign[:, l-1:N] = agg
                l_scores["simalign"] = sim_1_l_simalign

            # Fuse them
            fused_sim_1_l[l] = fuser.fuse(l_scores)

        print(f"[Aligner] Group similarity precomputation complete. ({__import__('time').time() - t_pre:.2f}s)")

        # ----------------------------------------------------
        # 3. Dynamic Programming Alignment
        # ----------------------------------------------------
        print("[Aligner] === Phase 1 complete. Running DP alignment... ===")
        t_dp = __import__('time').time()

        # Precompute k-1 scores in a single 3D array (shape: max_merge_han, M, N)
        # to allow fast vectorized inner loop over k.
        scores_k_1_3d = np.zeros((max_merge_han, M, N), dtype=np.float32)
        for k in range(1, max_merge_han + 1):
            matrix = fused_sim_k_1[k]
            scores_k_1_3d[k - 1] = np.where(matrix >= threshold, matrix - threshold, -10.0)

        # DP Helper functions to query precomputed fused similarity matrices
        def get_sim_k_1(i_start, i_end, j):
            k = i_end - i_start + 1
            return fused_sim_k_1[k][i_end, j]
            
        def get_sim_1_l(i, j_start, j_end):
            l = j_end - j_start + 1
            return fused_sim_1_l[l][i, j_end]

        # Initialize DP matrix
        dp = np.full((M + 1, N + 1), -1e9)
        ptr = np.zeros((M + 1, N + 1), dtype=object)
        dp[0][0] = 0.0

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
                if j > 0:
                    limit_k = min(i, max_merge_han)
                    if limit_k >= 1:
                        # Vectorized slice of dp values: [dp[i-1], dp[i-2], ..., dp[i-limit_k]]
                        dp_prevs = dp[i - limit_k : i, j - 1][::-1]
                        # Slice of precomputed 3D score matrix
                        scores = scores_k_1_3d[:limit_k, i - 1, j - 1]
                        
                        vals = dp_prevs + scores
                        best_k_idx = np.argmax(vals)
                        best_val = vals[best_k_idx]
                        
                        if best_val > dp[i][j]:
                            dp[i][j] = best_val
                            ptr[i][j] = (1, int(best_k_idx + 1), 1)
                            
                # Option 2: 1-l mapping (merge l Viet sentences to 1 Han sentence)
                limit_l = min(j, max_merge_viet)
                if limit_l >= 2 and i >= 1:
                    for l in range(2, limit_l + 1):
                        score = fused_sim_1_l[l][i - 1, j - 1]
                        score = score - threshold if score >= threshold else -10.0
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
                "similarity_score": round(float(sim), 4),
                "han_indices": h_idxs
            })
            idx += 1
            
        print(f"[Aligner] DP complete in {__import__('time').time() - t_dp:.2f}s. Found {len(results)} aligned pairs.")
        print(f"[Aligner] Total Phase 1 execution time: {__import__('time').time() - t_phase1:.2f}s")
        return results
