"""
Base interface for all sentence similarity scorers.
Each scorer must return a (M, N) numpy matrix with values in [0, 1].
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseScorer(ABC):
    """
    Abstract base class for pairwise sentence similarity scorers.

    All subclasses must implement `score()`, which takes two lists of sentences
    and returns a similarity matrix of shape (M, N) with values in [0, 1].
    """

    # Override in each subclass for logging purposes
    name: str = "base"

    @abstractmethod
    def score(self, han_sentences: List[str], viet_sentences: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity between Han and Viet sentences.

        Args:
            han_sentences: List of M Classical Chinese sentences.
            viet_sentences: List of N Vietnamese sentences.

        Returns:
            np.ndarray of shape (M, N), dtype float32, values clipped to [0, 1].
        """
        pass

    def is_available(self) -> bool:
        """
        Check if all required dependencies for this scorer are installed.
        Override in subclasses that have optional dependencies.
        """
        return True
