"""
Ensemble Fuser — Combines multiple similarity matrices with custom weights.

Supports dynamic weight redistribution for sparse scorers (like SimAlign).
If a scorer has no signal (value is 0) for a specific cell, its weight is
redistributed proportionally among the other active scorers for that cell.
"""

import time
from typing import Dict

import numpy as np


class EnsembleFuser:
    """
    Fuses multiple sentence similarity matrices using weighted average.

    Automatically normalizes weights to sum to 1.0.
    Handles sparse matrices by redistributing weight dynamically per-cell.
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dictionary mapping scorer name (lowercase) to its weight.
                     E.g., {"labse": 0.20, "vecalign": 0.30, ...}
        """
        # Ensure weights are positive
        self.raw_weights = {k: max(0.0, float(v)) for k, v in weights.items()}
        total = sum(self.raw_weights.values())
        if total == 0:
            raise ValueError("[Ensemble] Total weight sum cannot be 0.")

        # Normalize weights
        self.weights = {k: v / total for k, v in self.raw_weights.items()}

    def fuse(self, score_matrices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse multiple matrices.

        Args:
            score_matrices: Dict mapping scorer name (lowercase) to np.ndarray of shape (M, N).

        Returns:
            np.ndarray of shape (M, N), fused similarities in [0, 1].
        """
        # Verify all matrices have the same shape
        keys = list(score_matrices.keys())
        if not keys:
            raise ValueError("[Ensemble] No score matrices provided to fuse.")

        shape = score_matrices[keys[0]].shape
        for k in keys:
            if score_matrices[k].shape != shape:
                raise ValueError(
                    f"[Ensemble] Matrix shape mismatch. {keys[0]} has {shape}, but {k} has {score_matrices[k].shape}"
                )

        M, N = shape
        print(f"[Ensemble] Fusing {len(keys)} score matrices of shape {M}×{N}...")
        t0 = time.time()

        # Check weights configured vs matrices provided
        active_scorers = [k for k in keys if k in self.weights]
        if not active_scorers:
            raise ValueError(
                f"[Ensemble] None of the provided matrices {keys} match the configured weights {list(self.weights.keys())}"
            )

        # Standard weighted average if no sparse scorers (like simalign)
        # However, to be robust, we implement general dynamic cell-wise weight normalization.
        # Let's check if 'simalign' is in active scorers.
        has_simalign = "simalign" in active_scorers

        if not has_simalign:
            # Fast vectorized weighted average
            fused = np.zeros((M, N), dtype=np.float32)
            weight_sum = sum(self.weights[k] for k in active_scorers)
            for k in active_scorers:
                normalized_w = self.weights[k] / weight_sum
                fused += normalized_w * score_matrices[k]
        else:
            # SimAlign is sparse (values are 0 for non-computed pairs).
            # For each cell (i, j):
            # If simalign[i, j] > 0, we use normal weights:
            #   score = w_labse*labse + w_vecalign*vecalign + w_bert*bert + w_simalign*simalign
            # If simalign[i, j] == 0, we redistribute simalign's weight:
            #   score = (w_labse*labse + w_vecalign*vecalign + w_bert*bert) / (w_labse + w_vecalign + w_bert)
            #
            # We can vectorize this cell-wise redistribution:
            # Accumulate sum(score * weight) and sum(weight) cell-by-cell.

            weighted_sum = np.zeros((M, N), dtype=np.float32)
            weight_sum_matrix = np.zeros((M, N), dtype=np.float32)

            for k in active_scorers:
                matrix = score_matrices[k]
                w = self.weights[k]

                if k == "simalign":
                    # Only apply weight where simalign is non-zero
                    # For zero entries, simalign's weight is 0.0 for that cell
                    mask = (matrix > 0.0)
                    weighted_sum += np.where(mask, w * matrix, 0.0)
                    weight_sum_matrix += np.where(mask, w, 0.0)
                else:
                    # Dense scorer: weight always applies
                    weighted_sum += w * matrix
                    weight_sum_matrix += w

            # Avoid division by zero (should not happen since dense scorers are always active)
            weight_sum_matrix = np.where(weight_sum_matrix == 0.0, 1.0, weight_sum_matrix)
            fused = weighted_sum / weight_sum_matrix

        print(
            f"[Ensemble] Weights used: "
            + ", ".join([f"{k}={self.weights[k]:.3f}" for k in active_scorers])
        )
        print(f"[Ensemble] Done. ({time.time() - t0:.3f}s)")
        return np.clip(fused, 0.0, 1.0).astype(np.float32)
