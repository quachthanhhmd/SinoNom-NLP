import numpy as np
import pytest

from nlp.alignment_core import (
    AlignmentInvariantError,
    monotonic_span_align,
    validate_alignment_records,
)


def test_true_two_to_two_transition_is_available():
    # Cross similarity cannot be represented by monotonic 1-1 links, but the
    # combined contiguous 2-2 span has strong symmetric coverage.
    matrix = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
    records = monotonic_span_align(
        matrix,
        ["甲", "乙"],
        ["một", "hai"],
        max_merge_han=2,
        max_merge_viet=2,
        threshold=0.5,
        skip_penalty=0.5,
    )
    assert len(records) == 1
    assert records[0]["alignment_type"] == "2-2"
    assert records[0]["han_indices"] == [0, 1]
    assert records[0]["viet_indices"] == [0, 1]


def test_volume_boundary_cannot_be_crossed():
    records = monotonic_span_align(
        np.full((2, 2), 0.9, dtype=np.float32),
        ["卷一", "卷二"],
        ["quyển một", "quyển hai"],
        max_merge_han=2,
        max_merge_viet=2,
        threshold=0.5,
        han_breaks={1},
    )
    assert all(not ({0, 1} <= set(item["han_indices"])) for item in records)
    assert validate_alignment_records(records, 2, 2)["han_units"] == 2


def test_validator_rejects_duplicate_or_backtracking_indices():
    records = [
        {"han_indices": [0], "viet_indices": [0]},
        {"han_indices": [0, 1], "viet_indices": [1]},
    ]
    with pytest.raises(AlignmentInvariantError):
        validate_alignment_records(records)


def test_unmatched_records_still_preserve_complete_coverage():
    records = monotonic_span_align(
        np.zeros((2, 1), dtype=np.float32),
        ["甲", "乙"],
        ["không khớp"],
        threshold=0.8,
    )
    summary = validate_alignment_records(records, 2, 1)
    assert summary == {"records": 3, "han_units": 2, "viet_units": 1}
    assert {item["status"] for item in records} == {"unmatched"}
