"""
Vecalign Scorer — Context-window overlap embedding similarity.

Inspired by Vecalign (Thompson & Koehn, 2019).
Core idea: instead of comparing isolated sentence embeddings, each sentence is
represented by an *overlap embedding* — the average of its own embedding plus
its immediate neighbors. This makes the similarity signal robust to many-to-1
and 1-to-many alignments, because neighboring context "leaks" into each vector.

Example (window_size=2):
  overlap_han[i] = mean(han_embeds[max(0,i-2) : i+3])

This is fast: O(M+N) sliding-window averages using np.cumsum, no extra BERT calls.
"""

import time
from typing import List, Optional

import numpy as np

from .base import BaseScorer


class VecalignScorer(BaseScorer):
    """
    Context-aware scorer using overlap embeddings (Vecalign-inspired).

    Reuses pre-encoded embeddings from LaBSEScorer to avoid double-encoding.
    Adds context by averaging neighboring sentence embeddings within a window.
    """

    name = "Vecalign"

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Number of neighboring sentences on each side to include
                         in the overlap embedding. E.g., window_size=3 means each
                         sentence's embedding is averaged with up to 3 neighbors
                         on each side (total window: up to 7 sentences).
        """
        self.window_size = window_size

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _overlap_embeddings(self, embeds_norm: np.ndarray) -> np.ndarray:
        """
        Compute overlap (context-averaged) embeddings using np.cumsum.

        For sentence i, overlap[i] = mean of embeds[max(0,i-W) : i+W+1],
        then L2-normalized.

        Returns:
            np.ndarray of shape (N, D), L2-normalized.
        """
        N, D = embeds_norm.shape
        W = self.window_size

        # Padded cumsum for efficient window summation
        # padded[k] = sum of embeds[0..k-1]
        padded = np.vstack([np.zeros((1, D)), np.cumsum(embeds_norm, axis=0)])  # (N+1, D)

        # For each i: start=max(0, i-W), end=min(N, i+W+1)
        starts = np.maximum(0, np.arange(N) - W)
        ends = np.minimum(N, np.arange(N) + W + 1)
        counts = (ends - starts).reshape(-1, 1).astype(np.float32)  # (N, 1)

        # Sum over window using padded cumsum
        sums = padded[ends] - padded[starts]  # (N, D)
        means = sums / counts  # (N, D)

        # L2-normalize
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (means / norms).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(
        self,
        han_sentences: List[str],
        viet_sentences: List[str],
        han_embeds_norm: Optional[np.ndarray] = None,
        viet_embeds_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute context-aware cosine similarity matrix.

        Args:
            han_sentences:      List of M Han sentences (used for length only if embeds given).
            viet_sentences:     List of N Viet sentences.
            han_embeds_norm:    L2-normalized LaBSE embeddings (M, D). Required.
            viet_embeds_norm:   L2-normalized LaBSE embeddings (N, D). Required.

        Returns:
            np.ndarray of shape (M, N), values in [0, 1].
        """
        if han_embeds_norm is None or viet_embeds_norm is None:
            raise ValueError(
                f"[{self.name}] VecalignScorer requires pre-encoded embeddings. "
                "Pass han_embeds_norm and viet_embeds_norm from LaBSEScorer."
            )

        M, N = len(han_sentences), len(viet_sentences)
        print(f"[{self.name}] Computing overlap embeddings (window_size={self.window_size})...")
        t0 = time.time()

        han_overlap = self._overlap_embeddings(han_embeds_norm)   # (M, D)
        viet_overlap = self._overlap_embeddings(viet_embeds_norm)  # (N, D)

        print(f"[{self.name}] Overlap embeddings done. Computing similarity matrix ({M}×{N})...")
        sim = han_overlap @ viet_overlap.T  # (M, N)
        print(f"[{self.name}] Done. ({time.time() - t0:.3f}s)")
        return np.clip(sim, 0.0, 1.0).astype(np.float32)
