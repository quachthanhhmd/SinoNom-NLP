import os
import tempfile

import pandas as pd

from nlp.bead_quality import (
    split_non_exact_for_repair,
    teacher_sample_result,
)
from nlp.qwen_verifier import QwenVerifier
from utils.exporters import CorpusExporter


def test_verifier_parses_strict_completeness_label():
    result = QwenVerifier._parse_result(
        '{"label":"addition","extra_side":"viet","missing_side":"none",'
        '"confidence":0.91,"reason":"Việt có thêm một địa danh."}'
    )
    assert result["label"] == "addition"
    assert result["extra_side"] == "viet"
    assert result["confidence"] == 0.91


def test_malformed_verifier_output_fails_closed():
    result = QwenVerifier._parse_result("không thể quyết định")
    assert result["label"] == "mismatch"
    assert result["confidence"] == 0.0


def test_non_exact_merged_bead_is_restored_to_atomic_source_units():
    records = [{
        "han_sentence": "H0 H1",
        "viet_sentence": "V0 V1",
        "han_indices": [0, 1],
        "viet_indices": [0, 1],
        "completeness_label": "omission",
        "status": "omission",
    }]

    routed, rejected = split_non_exact_for_repair(
        records,
        repair_round=0,
        han_source_units=["H0", "H1"],
        viet_source_units=["V0", "V1"],
    )

    assert [item["han_indices"] for item in routed if item["han_indices"]] == [[0], [1]]
    assert [item["viet_indices"] for item in routed if item["viet_indices"]] == [[0], [1]]
    assert len(rejected) == 1
    assert rejected[0]["completeness_label"] == "omission"


def test_exact_bead_remains_an_immutable_anchor():
    record = {
        "han_sentence": "H0 H1",
        "viet_sentence": "V0",
        "han_indices": [0, 1],
        "viet_indices": [0],
        "completeness_label": "exact",
    }
    routed, rejected = split_non_exact_for_repair(
        [record],
        repair_round=1,
        han_source_units=["H0", "H1"],
        viet_source_units=["V0"],
    )
    assert routed == [record]
    assert routed[0]["status"] == "accepted"
    assert rejected == []


def test_teacher_rule_fails_only_above_one_third_errors():
    assert teacher_sample_result(["exact"] * 6 + ["addition"] * 3)["passed"]
    assert not teacher_sample_result(["exact"] * 5 + ["omission"] * 4)["passed"]


def test_exporter_never_promotes_similarity_only_match_to_exact():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as output_dir:
        exporter = CorpusExporter(output_dir)
        exporter.export_hierarchical(
            parent_dir="BOOK",
            chapter_str="01",
            aligned_data=[
                {
                    "pair_id": "exact",
                    "han_sentence": "甲",
                    "viet_sentence": "Giáp",
                    "status": "accepted",
                    "completeness_label": "exact",
                },
                {
                    "pair_id": "unsafe",
                    "han_sentence": "乙",
                    "viet_sentence": "Ất và nội dung thêm",
                    "status": "accepted",
                    "completeness_label": "addition",
                },
            ],
        )
        path = os.path.join(
            output_dir, "BOOK", "BOOK_01", "BOOK_01_exact_accepted.tsv"
        )
        exported = pd.read_csv(path, sep="\t")
        assert exported["pair_id"].tolist() == ["exact"]
