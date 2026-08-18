"""Quality gates and repair routing for Hán--Việt alignment beads.

The structural aligner guarantees monotonic source coverage.  This module adds
the stricter, content-level contract used by the project evaluator: a training
bead is accepted only when both sides contain the same complete information.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


COMPLETENESS_LABELS = {"exact", "addition", "omission", "mismatch"}
NON_EXACT_LABELS = COMPLETENESS_LABELS - {"exact"}


def has_text(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() != "nan"


def normalize_label(value: Any) -> str:
    """Normalize an LLM label; invalid output fails closed as ``mismatch``."""
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "match": "exact",
        "matched": "exact",
        "correct": "exact",
        "fully_equivalent": "exact",
        "extra": "addition",
        "added": "addition",
        "missing": "omission",
        "omit": "omission",
        "wrong": "mismatch",
        "unrelated": "mismatch",
        "partial": "mismatch",
    }
    label = aliases.get(label, label)
    return label if label in COMPLETENESS_LABELS else "mismatch"


def bead_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    """Stable checkpoint key that does not collide on repeated sentence text."""
    return (
        tuple(item.get("han_indices", [])),
        tuple(item.get("viet_indices", [])),
        str(item.get("han_sentence", "")),
        str(item.get("viet_sentence", "")),
    )


def mark_verified_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the exact-only acceptance policy without changing source coverage."""
    for item in records:
        both_sides = has_text(item.get("han_sentence")) and has_text(item.get("viet_sentence"))
        if not both_sides:
            item["completeness_label"] = "unmatched"
            item["verified"] = False
            item["status"] = "unmatched"
            continue

        label = normalize_label(item.get("completeness_label"))
        item["completeness_label"] = label
        item["verified"] = label == "exact"
        item["status"] = "accepted" if label == "exact" else label
    return records


def split_non_exact_for_repair(
    records: List[Dict[str, Any]],
    repair_round: int,
    han_source_units: List[str] | None = None,
    viet_source_units: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Turn rejected two-sided beads back into traceable one-sided source units.

    Exact beads remain immutable anchors.  Addition/omission/mismatch beads are
    split so a local monotonic realigner can redraw their boundaries.  A copy of
    each rejected candidate is returned for diagnostics and separate exports.
    """
    routed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for item in mark_verified_records(records):
        label = item.get("completeness_label")
        both_sides = has_text(item.get("han_sentence")) and has_text(item.get("viet_sentence"))
        if not both_sides or label == "exact":
            routed.append(item)
            continue

        rejected_item = dict(item)
        rejected_item["repair_round"] = repair_round
        rejected.append(rejected_item)

        common = {
            "similarity_score": 0.0,
            "confidence": 0.0,
            "status": "unmatched",
            "verified": False,
            "completeness_label": "unmatched",
            "repair_reason": label,
            "repair_round": repair_round,
            "rejected_han_sentence": item.get("han_sentence", ""),
            "rejected_viet_sentence": item.get("viet_sentence", ""),
            "rejected_extra_side": item.get("extra_side", "none"),
            "rejected_missing_side": item.get("missing_side", "none"),
        }

        han_indices = list(item.get("han_indices", []))
        viet_indices = list(item.get("viet_indices", []))

        # Restore atomic source rows before drawing a new m-n boundary. Without
        # this step a rejected 2-1 bead would remain fused forever and could
        # never become two exact beads (or a different 1-1/1-2 combination).
        if has_text(item.get("han_sentence")):
            if han_source_units is not None and han_indices:
                han_parts = [([index], han_source_units[index]) for index in han_indices]
            else:
                han_parts = [(han_indices, item.get("han_sentence", ""))]
            for indices, sentence in han_parts:
                routed.append({
                    **common,
                    "han_sentence": sentence,
                    "viet_sentence": "",
                    "han_indices": indices,
                    "viet_indices": [],
                })
        if has_text(item.get("viet_sentence")):
            if viet_source_units is not None and viet_indices:
                viet_parts = [([index], viet_source_units[index]) for index in viet_indices]
            else:
                viet_parts = [(viet_indices, item.get("viet_sentence", ""))]
            for indices, sentence in viet_parts:
                routed.append({
                    **common,
                    "han_sentence": "",
                    "viet_sentence": sentence,
                    "han_indices": [],
                    "viet_indices": indices,
                })

    return routed, rejected


def quality_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for item in records:
        if not has_text(item.get("han_sentence")) or not has_text(item.get("viet_sentence")):
            counts["unmatched"] += 1
        else:
            counts[normalize_label(item.get("completeness_label"))] += 1
    return dict(counts)


def teacher_sample_result(labels: Iterable[str]) -> Dict[str, Any]:
    """Apply the teacher's rule: fail only when errors exceed one third."""
    normalized = [normalize_label(label) for label in labels]
    total = len(normalized)
    errors = sum(label != "exact" for label in normalized)
    error_rate = errors / total if total else None
    return {
        "sample_size": total,
        "errors": errors,
        "error_rate": error_rate,
        "threshold": 1 / 3,
        "passed": bool(total) and errors <= total / 3,
    }
