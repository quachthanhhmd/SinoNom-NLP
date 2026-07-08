"""
BERTAlign Scorer — Multilingual sentence similarity with a different model from LaBSE.

Uses `paraphrase-multilingual-MiniLM-L12-v2`, which was trained with a paraphrase
objective (different from LaBSE's parallel-corpus objective). This gives a genuinely
diverse signal: the two models disagree on borderline cases in complementary ways,
which ensemble averaging exploits.

Why not use the bertalign library directly?
  bertalign (the PyPI package) returns aligned sentence pairs, not a similarity
  matrix — it's a full alignment pipeline, not a scorer. We replicate its core
  idea (BERT-based multilingual embeddings + cosine similarity) with a model that
  is openly available on HuggingFace and works well for ZH→VI alignment.
"""

import time
from typing import List, Optional

import numpy as np

from .base import BaseScorer


class BERTAlignScorer(BaseScorer):
    """
    Scorer using paraphrase-multilingual-MiniLM-L12-v2.

    Trained on 50+ languages with a paraphrase objective. Provides diverse signal
    compared to LaBSE (which uses a bitext-mining objective).
    """

    name = "BERTAlign"

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    # ------------------------------------------------------------------ #
    # Properties
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _encode_normalized(self, sentences: List[str], label: str) -> np.ndarray:
        print(f"[{self.name}] Encoding {len(sentences)} {label}...")
        t0 = time.time()
        embeds = self.model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=256,
        )
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        print(f"[{self.name}] Encoding done. ({time.time() - t0:.2f}s)")
        return (embeds / norms).astype(np.float32)

    def score(
        self,
        han_sentences: List[str],
        viet_sentences: List[str],
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix using paraphrase-multilingual-MiniLM.

        Returns:
            np.ndarray of shape (M, N), values in [0, 1].
        """
        han_norm = self._encode_normalized(han_sentences, "Han sentences")
        viet_norm = self._encode_normalized(viet_sentences, "Viet sentences")

        M, N = len(han_sentences), len(viet_sentences)
        print(f"[{self.name}] Computing similarity matrix ({M}×{N})...")
        t0 = time.time()
        sim = han_norm @ viet_norm.T
        print(f"[{self.name}] Done. ({time.time() - t0:.3f}s)")
        return np.clip(sim, 0.0, 1.0).astype(np.float32)
