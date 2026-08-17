"""Deterministic monotonic m-n alignment and invariant validation.

This module is intentionally model-agnostic.  Scorers produce a base MxN
similarity matrix; the decoder then chooses monotonic contiguous spans while
preserving every source unit exactly once (matched or unmatched).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class AlignmentInvariantError(ValueError):
    """Raised when an alignment reuses, drops, or reorders source units."""


def _as_index_list(value: Any, field: str) -> List[int]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise AlignmentInvariantError(f"{field} must be a list of integers: {value!r}")
    if value != sorted(value) or len(value) != len(set(value)):
        raise AlignmentInvariantError(f"{field} must be unique and increasing: {value!r}")
    if value and value != list(range(value[0], value[-1] + 1)):
        raise AlignmentInvariantError(f"{field} must describe one contiguous span: {value!r}")
    return value


def validate_alignment_records(
    records: Sequence[Dict[str, Any]],
    expected_han_count: Optional[int] = None,
    expected_viet_count: Optional[int] = None,
) -> Dict[str, int]:
    """Validate monotonicity, uniqueness, and optional complete coverage."""

    seen_han: List[int] = []
    seen_viet: List[int] = []
    previous_han = -1
    previous_viet = -1

    for position, record in enumerate(records):
        han_indices = _as_index_list(record.get("han_indices", []), "han_indices")
        viet_indices = _as_index_list(record.get("viet_indices", []), "viet_indices")
        if not han_indices and not viet_indices:
            raise AlignmentInvariantError(f"record {position} has neither Han nor Viet indices")
        if han_indices and han_indices[0] <= previous_han:
            raise AlignmentInvariantError(
                f"Han indices overlap/backtrack at record {position}: {han_indices!r}"
            )
        if viet_indices and viet_indices[0] <= previous_viet:
            raise AlignmentInvariantError(
                f"Viet indices overlap/backtrack at record {position}: {viet_indices!r}"
            )
        if han_indices:
            previous_han = han_indices[-1]
            seen_han.extend(han_indices)
        if viet_indices:
            previous_viet = viet_indices[-1]
            seen_viet.extend(viet_indices)

    if len(seen_han) != len(set(seen_han)):
        raise AlignmentInvariantError("Han source units are reused")
    if len(seen_viet) != len(set(seen_viet)):
        raise AlignmentInvariantError("Viet source units are reused")
    if expected_han_count is not None and seen_han != list(range(expected_han_count)):
        raise AlignmentInvariantError(
            f"Han coverage mismatch: expected {expected_han_count}, got {len(seen_han)}"
        )
    if expected_viet_count is not None and seen_viet != list(range(expected_viet_count)):
        raise AlignmentInvariantError(
            f"Viet coverage mismatch: expected {expected_viet_count}, got {len(seen_viet)}"
        )

    return {
        "records": len(records),
        "han_units": len(seen_han),
        "viet_units": len(seen_viet),
    }


def _span_semantic_score(matrix: np.ndarray) -> float:
    """Symmetric coverage score for an m-by-n base-similarity submatrix."""

    if matrix.size == 0:
        return 0.0
    row_coverage = float(np.mean(np.max(matrix, axis=1)))
    col_coverage = float(np.mean(np.max(matrix, axis=0)))
    return (row_coverage + col_coverage) / 2.0


def _length_score(han_text: str, viet_text: str, target_ratio: float = 1.4) -> float:
    han_chars = max(1, sum(1 for char in han_text if not char.isspace()))
    viet_words = max(1, len(viet_text.split()))
    observed = viet_words / han_chars
    return math.exp(-abs(math.log(max(observed, 1e-6) / target_ratio)))


def monotonic_span_align(
    similarity_matrix: np.ndarray,
    han_sentences: Sequence[str],
    viet_sentences: Sequence[str],
    *,
    max_merge_han: int = 3,
    max_merge_viet: int = 3,
    threshold: float = 0.32,
    skip_penalty: float = 0.18,
    merge_penalty: float = 0.04,
    length_weight: float = 0.08,
    han_breaks: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Decode a true monotonic m-n path over contiguous source spans.

    Supported transitions are every (m, n) where 1 <= m <= max_merge_han and
    1 <= n <= max_merge_viet, plus one-sided skips.  Each source unit is
    consumed exactly once, so a local boundary error cannot duplicate or drop
    material silently.
    """

    matrix = np.asarray(similarity_matrix, dtype=np.float32)
    m_total, n_total = len(han_sentences), len(viet_sentences)
    if matrix.shape != (m_total, n_total):
        raise ValueError(
            f"similarity matrix shape {matrix.shape} does not match {(m_total, n_total)}"
        )
    if max_merge_han < 1 or max_merge_viet < 1:
        raise ValueError("max merge sizes must be positive")
    protected_han_breaks = set(han_breaks or [])
    if any(boundary <= 0 or boundary >= m_total for boundary in protected_han_breaks):
        raise ValueError(f"invalid Han boundaries: {sorted(protected_han_breaks)}")

    negative_infinity = -1e18
    dp = np.full((m_total + 1, n_total + 1), negative_infinity, dtype=np.float64)
    pointers: List[List[Optional[Tuple[int, int, float, float]]]] = [
        [None] * (n_total + 1) for _ in range(m_total + 1)
    ]
    dp[0, 0] = 0.0

    for i in range(m_total + 1):
        for j in range(n_total + 1):
            current = dp[i, j]
            if current <= negative_infinity / 2:
                continue

            if i < m_total:
                candidate = current - skip_penalty
                if candidate > dp[i + 1, j]:
                    dp[i + 1, j] = candidate
                    pointers[i + 1][j] = (1, 0, 0.0, -skip_penalty)

            if j < n_total:
                candidate = current - skip_penalty
                if candidate > dp[i, j + 1]:
                    dp[i, j + 1] = candidate
                    pointers[i][j + 1] = (0, 1, 0.0, -skip_penalty)

            for han_size in range(1, min(max_merge_han, m_total - i) + 1):
                if any(i < boundary < i + han_size for boundary in protected_han_breaks):
                    continue
                for viet_size in range(1, min(max_merge_viet, n_total - j) + 1):
                    semantic = _span_semantic_score(
                        matrix[i : i + han_size, j : j + viet_size]
                    )
                    if semantic < threshold:
                        continue
                    han_text = " ".join(han_sentences[i : i + han_size])
                    viet_text = " ".join(viet_sentences[j : j + viet_size])
                    length_bonus = length_weight * (_length_score(han_text, viet_text) - 0.5)
                    complexity_penalty = merge_penalty * (han_size + viet_size - 2)
                    transition_score = semantic - threshold + length_bonus - complexity_penalty
                    next_i, next_j = i + han_size, j + viet_size
                    candidate = current + transition_score
                    if candidate > dp[next_i, next_j]:
                        dp[next_i, next_j] = candidate
                        pointers[next_i][next_j] = (
                            han_size,
                            viet_size,
                            semantic,
                            transition_score,
                        )

    i, j = m_total, n_total
    reversed_records: List[Dict[str, Any]] = []
    while i > 0 or j > 0:
        pointer = pointers[i][j]
        if pointer is None:
            raise AlignmentInvariantError(f"decoder cannot backtrack state {(i, j)}")
        han_size, viet_size, semantic, transition_score = pointer
        start_i, start_j = i - han_size, j - viet_size
        han_indices = list(range(start_i, i)) if han_size else []
        viet_indices = list(range(start_j, j)) if viet_size else []
        han_text = " ".join(han_sentences[index] for index in han_indices)
        viet_text = " ".join(viet_sentences[index] for index in viet_indices)
        if han_indices and viet_indices:
            status = "accepted" if semantic >= max(0.50, threshold + 0.12) else "review"
        else:
            status = "unmatched"
        reversed_records.append(
            {
                "han_sentence": han_text,
                "viet_sentence": viet_text,
                "han_indices": han_indices,
                "viet_indices": viet_indices,
                "alignment_type": f"{han_size}-{viet_size}",
                "similarity_score": round(float(semantic), 4),
                "alignment_score": round(float(transition_score), 4),
                "confidence": round(float(semantic), 4) if han_indices and viet_indices else 0.0,
                "status": status,
            }
        )
        i, j = start_i, start_j

    records = list(reversed(reversed_records))
    for pair_number, record in enumerate(records, start=1):
        record["pair_id"] = f"pair_{pair_number:06d}"
    validate_alignment_records(records, m_total, n_total)
    return records
