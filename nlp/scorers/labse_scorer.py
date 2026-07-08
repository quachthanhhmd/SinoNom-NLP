"""
LaBSE Scorer — Language-Agnostic BERT Sentence Embeddings.

Computes cosine similarity between independently encoded Han and Viet sentences.
Fast bi-encoder: O(M+N) encode calls, O(M*N) dot products.
"""

import time
from typing import List, Optional

import numpy as np

from .base import BaseScorer


class LaBSEScorer(BaseScorer):
    """
    Scorer using LaBSE (sentence-transformers/LaBSE).

    Encodes Han and Viet sentences independently, then computes the full
    cosine similarity matrix via matrix multiplication.

    Supports sharing pre-computed embeddings from outside to avoid
    double-encoding when multiple scorers use the same model.
    """

    name = "LaBSE"

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def model(self):
        if self._model is None:
            print(f"[{self.name}] Loading model: {self.model_name}...")
            t0 = time.time()
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[{self.name}] Model loaded. ({time.time() - t0:.1f}s)")
        return self._model

    def encode(self, sentences: List[str], label: str = "sentences") -> np.ndarray:
        """
        Encode a list of sentences. Returns normalized embeddings (L2).
        Exposed publicly so other scorers can reuse these embeddings.
        """
        print(f"[{self.name}] Encoding {len(sentences)} {label}...")
        t0 = time.time()
        embeds = self.model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeds_norm = embeds / norms
        print(f"[{self.name}] Encoding done. ({time.time() - t0:.2f}s)")
        return embeds_norm  # shape: (N, D)

    def score(
        self,
        han_sentences: List[str],
        viet_sentences: List[str],
        han_embeds_norm: Optional[np.ndarray] = None,
        viet_embeds_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix.

        Args:
            han_sentences:      List of M Han sentences.
            viet_sentences:     List of N Viet sentences.
            han_embeds_norm:    Optional pre-encoded, L2-normalized Han embeddings (M, D).
            viet_embeds_norm:   Optional pre-encoded, L2-normalized Viet embeddings (N, D).

        Returns:
            np.ndarray of shape (M, N), values in [0, 1].
        """
        if han_embeds_norm is None:
            han_embeds_norm = self.encode(han_sentences, label="Han sentences")
        if viet_embeds_norm is None:
            viet_embeds_norm = self.encode(viet_sentences, label="Viet sentences")

        M, N = len(han_sentences), len(viet_sentences)
        print(f"[{self.name}] Computing similarity matrix ({M}×{N})...")
        t0 = time.time()
        sim = han_embeds_norm @ viet_embeds_norm.T  # (M, N)
        print(f"[{self.name}] Done. ({time.time() - t0:.3f}s)")
        return np.clip(sim, 0.0, 1.0).astype(np.float32)
