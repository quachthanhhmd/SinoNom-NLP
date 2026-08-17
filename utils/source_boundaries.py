"""Pure helpers for deterministic source ordering and volume boundaries."""

import re
from typing import Dict, Iterable, List


def natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def split_merged_sino_rows(
    rows: Iterable[dict], volume_codes: List[str]
) -> Dict[str, List[dict]]:
    """Split a two-volume OCR stream using verified headings in its text."""
    if len(volume_codes) != 2:
        raise ValueError("merged Sino split currently requires exactly two ordered volumes")
    first_volume, second_volume = volume_codes
    marker_text = {11: "十一", 17: "十七"}.get(int(second_volume))
    if marker_text is None:
        raise ValueError(f"no heading marker configured for volume {second_volume}")
    marker = re.compile(rf"大南一統志卷之?{marker_text}|卷之?{marker_text}")
    result = {first_volume: [], second_volume: []}
    active_volume = first_volume
    boundary_found = False

    for row in rows:
        original_id = str(row.get("ID", "")).strip()
        sentence = str(row.get("sentence", "")).strip()
        if not sentence:
            continue
        match = marker.search(sentence)
        if match and active_volume == first_volume:
            before = sentence[:match.start()].strip()
            if before:
                result[first_volume].append({"ID": original_id, "sentence": before})
            active_volume = second_volume
            boundary_found = True
            sentence = sentence[match.start():].strip()
        result[active_volume].append({"ID": original_id, "sentence": sentence})

    if not boundary_found or not result[first_volume] or not result[second_volume]:
        raise ValueError(
            f"cannot verify content heading boundary {first_volume}->{second_volume}; "
            "alignment is blocked instead of silently assigning the wrong volume"
        )
    return result
