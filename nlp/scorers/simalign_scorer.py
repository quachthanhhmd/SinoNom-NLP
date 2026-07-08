"""
SimAlign Scorer — Word-level alignment similarity.

Uses contextualized embeddings (XLM-R by default) via the `simalign` library.
Core idea: align individual words, and rate sentence similarity by the ratio:
  score = (number of aligned word pairs) / max(len(han_tokens), len(viet_tokens))

Since word-level alignment requires O(M*N) calls to BERT/XLM-R token embeddings,
computing this for a full book would be extremely slow.
To optimize, we use a *Sparse Scorer* approach:
  - We accept a `reference_matrix` (e.g. from LaBSEScorer).
  - For each Han sentence, we only run SimAlign for the top K (e.g., K=5)
    Viet sentences matching that Han sentence.
  - All other entries are left as 0.0. The EnsembleFuser will redistribute weight
    for these zero-score cells.
"""

import time
from typing import List, Optional

import numpy as np

from .base import BaseScorer


class SimAlignScorer(BaseScorer):
    """
    Sparse scorer using SimAlign for word-level matching.

    Reuses token embeddings from XLM-R to find cross-lingual word alignments.
    To avoid performance issues, it only runs on top-K candidate pairs from a
    reference similarity matrix.
    """

    name = "SimAlign"

    def __init__(self, model_name: str = "xlmr", top_k: int = 5):
        """
        Args:
            model_name: "xlmr" (default), "bert", or path/name of HF model.
            top_k: Number of candidate Viet sentences per Han sentence to score.
        """
        self.model_name = model_name
        self.top_k = top_k
        self._aligner = None

    def is_available(self) -> bool:
        try:
            import simalign
            return True
        except ImportError:
            return False

    @property
    def aligner(self):
        if self._aligner is None:
            if not self.is_available():
                raise ImportError(
                    f"[{self.name}] simalign is not installed. Please run: pip install simalign"
                )
            print(f"[{self.name}] Loading SimAligner model ({self.model_name})...")
            t0 = time.time()
            from simalign import SentenceAligner
            # Disable printing inside simalign initializer if possible, or let it load
            self._aligner = SentenceAligner(model=self.model_name, token_type="bpe", device="cuda" if self._aligner_device_is_cuda() else "cpu")
            print(f"[{self.name}] Model loaded. ({time.time() - t0:.1f}s)")
        return self._aligner

    def _aligner_device_is_cuda(self) -> bool:
        # Check if cuda is available
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def score(
        self,
        han_sentences: List[str],
        viet_sentences: List[str],
        reference_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute sparse similarity matrix.

        Args:
            han_sentences: List of M Han sentences.
            viet_sentences: List of N Viet sentences.
            reference_matrix: Matrix (M, N) of candidate scores. If None,
                              we compute top_k on the entire N candidates (very slow!).

        Returns:
            np.ndarray of shape (M, N), sparse, values in [0, 1].
        """
        M, N = len(han_sentences), len(viet_sentences)
        scores = np.zeros((M, N), dtype=np.float32)

        if not self.is_available():
            print(f"[{self.name}] Warning: simalign package not installed. Skipping scorer (returns zeros).")
            return scores

        if reference_matrix is None:
            print(f"[{self.name}] Warning: No reference matrix provided to SimAlignScorer. SimAlign will be extremely slow.")
            # Dummy reference matrix where all values are equal
            reference_matrix = np.ones((M, N), dtype=np.float32)

        print(f"[{self.name}] Computing word-level alignments for top-{self.top_k} candidates...")
        t0 = time.time()

        # Instantiate model lazily
        aligner_instance = self.aligner
        align_count = 0

        # Run SimAlign on top-K candidate indices
        for i in range(M):
            # Find indices of top K highest similarity scores in reference matrix
            row_scores = reference_matrix[i]
            # argsort sorts ascending, so we take the last top_k elements and reverse
            top_k_indices = np.argsort(row_scores)[-self.top_k:][::-1]

            han_sent = han_sentences[i]
            if not han_sent.strip():
                continue

            for j in top_k_indices:
                viet_sent = viet_sentences[j]
                if not viet_sent.strip():
                    continue

                # Get word alignments
                try:
                    # SimAlign's get_word_aligns returns dict with keys: 'mwmf', 'itermax', 'match'
                    # We use 'itermax' or 'mwmf' as they are standard. 'mwmf' is usually higher quality.
                    alignments = aligner_instance.get_word_aligns(han_sent, viet_sent)
                    align_pairs = alignments.get("mwmf", alignments.get("itermax", []))

                    # Parse tokens to get length
                    # simalign.SentenceAligner has internal tokenizer
                    # We can estimate token count or use character tokenization fallback
                    han_tokens = han_sent.split()
                    viet_tokens = viet_sent.split()

                    max_tokens = max(len(han_tokens), len(viet_tokens))
                    if max_tokens > 0:
                        sim_score = len(align_pairs) / max_tokens
                        scores[i, j] = min(sim_score, 1.0)
                    else:
                        scores[i, j] = 0.0

                    align_count += 1
                except Exception as e:
                    # Handle tokenization errors or out-of-vocabulary gracefully
                    scores[i, j] = 0.0

        print(f"[{self.name}] Computed {align_count} sentence pairs. Done. ({time.time() - t0:.2f}s)")
        return scores
