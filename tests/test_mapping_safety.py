import pytest

from nlp.segmenter import RegexSegmenter
from utils.source_boundaries import natural_sort_key, split_merged_sino_rows


def test_merged_volume_is_split_by_heading_not_id_prefix():
    source = [
            {"ID": "Q10_001", "sentence": "卷十內容"},
            {"ID": "Q10_002", "sentence": "卷十末大南一統志卷之十一卷十一內容"},
            {"ID": "Q10_003", "sentence": "卷十一續"},
        ]
    result = split_merged_sino_rows(source, ["10", "11"])
    assert list(result) == ["10", "11"]
    assert "卷十末" in "".join(row["sentence"] for row in result["10"])
    assert result["11"][0]["sentence"].startswith("大南一統志卷之十一")


def test_merged_volume_without_verified_heading_is_blocked():
    source = [{"ID": "Q10_001", "sentence": "只有卷十"}]
    with pytest.raises(ValueError, match="cannot verify content heading boundary"):
        split_merged_sino_rows(source, ["10", "11"])


def test_ocr_pages_use_natural_order():
    pages = ["page_10.png", "page_2.png", "page_1.png"]
    assert sorted(pages, key=natural_sort_key) == [
        "page_1.png", "page_2.png", "page_10.png"
    ]


def test_single_newline_is_soft_but_paragraph_break_is_preserved():
    segmenter = RegexSegmenter(lang="han")
    assert segmenter.segment("天地\n玄黃。宇宙\n\n洪荒。") == [
        "天地 玄黃。", "宇宙", "洪荒。"
    ]
