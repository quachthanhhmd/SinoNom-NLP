"""Utilities for Task 2: Vietnamese PDF extraction, cleaning, and sentence splitting.

This module is intentionally independent from OCR and alignment components.
It can be used from a notebook, from scripts, or from the original pipeline.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class PDFPageText:
    page: int
    text: str


@dataclass
class SentenceRecord:
    work_id: str
    lang: str
    page: int
    sent_id: int
    text: str


def normalize_text(text: str) -> str:
    """Normalize Unicode and common PDF punctuation artifacts."""
    text = unicodedata.normalize("NFC", text or "")
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_pdf_pages(pdf_path: str | Path, start_page: int = 1, end_page: Optional[int] = None) -> List[PDFPageText]:
    """Extract text page by page from a text-based PDF.

    Page numbers are 1-based. The function first tries PyMuPDF because it is
    usually more stable for layout-heavy PDFs, then falls back to pypdf/PyPDF2.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    start_page = max(1, int(start_page or 1))
    pages: List[PDFPageText] = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        last_page = total_pages if end_page is None else min(total_pages, int(end_page))
        for page_no in range(start_page, last_page + 1):
            page = doc.load_page(page_no - 1)
            text = page.get_text("text") or ""
            pages.append(PDFPageText(page=page_no, text=normalize_text(text)))
        doc.close()
        return pages
    except ImportError:
        pass

    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        last_page = total_pages if end_page is None else min(total_pages, int(end_page))
        for page_no in range(start_page, last_page + 1):
            text = reader.pages[page_no - 1].extract_text() or ""
            pages.append(PDFPageText(page=page_no, text=normalize_text(text)))
        return pages
    except ImportError as exc:
        raise ImportError("Install pymupdf or pypdf to extract PDF text: pip install pymupdf pypdf") from exc


def _line_key(line: str) -> str:
    line = normalize_text(line).strip().lower()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\d+", "#", line)
    return line


def detect_repeated_lines(
    pages: Sequence[PDFPageText],
    threshold_ratio: float = 0.25,
    min_occurrences: int = 3,
    max_chars: int = 100,
) -> set[str]:
    """Detect likely headers/footers repeated across many pages."""
    if not pages:
        return set()

    per_page_keys: List[set[str]] = []
    for page in pages:
        keys = set()
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line or len(line) > max_chars:
                continue
            key = _line_key(line)
            if key:
                keys.add(key)
        per_page_keys.append(keys)

    counts: Counter[str] = Counter()
    for keys in per_page_keys:
        counts.update(keys)

    threshold = max(min_occurrences, int(len(pages) * threshold_ratio))
    return {key for key, cnt in counts.items() if cnt >= threshold}


def _looks_like_page_number(line: str) -> bool:
    line = line.strip()
    if re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", line):
        return True
    if re.fullmatch(r"trang\s+\d{1,4}", line, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_noise(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    if line.startswith("file://") or "file:///" in line:
        return True
    if re.search(r"https?://\S+", line):
        return True
    if re.fullmatch(r"[\W_]+", line, flags=re.UNICODE):
        return True
    return False


def clean_vietnamese_pages(
    pages: Sequence[PDFPageText],
    remove_repeated: bool = True,
    repeated_threshold_ratio: float = 0.25,
    min_line_chars: int = 2,
    custom_drop_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[PDFPageText], Dict[str, Any]]:
    """Clean extracted Vietnamese PDF text while preserving page metadata."""
    repeated_keys = detect_repeated_lines(pages, threshold_ratio=repeated_threshold_ratio) if remove_repeated else set()
    compiled_custom = [re.compile(p, flags=re.IGNORECASE) for p in (custom_drop_patterns or [])]

    cleaned_pages: List[PDFPageText] = []
    stats = {
        "input_pages": len(pages),
        "removed_empty_or_noise_lines": 0,
        "removed_page_number_lines": 0,
        "removed_repeated_lines": 0,
        "removed_custom_pattern_lines": 0,
        "kept_lines": 0,
        "repeated_keys": sorted(repeated_keys),
    }

    for page in pages:
        kept: List[str] = []
        for raw_line in normalize_text(page.text).splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if len(line) < min_line_chars or _looks_like_noise(line):
                stats["removed_empty_or_noise_lines"] += 1
                continue
            if _looks_like_page_number(line):
                stats["removed_page_number_lines"] += 1
                continue
            if _line_key(line) in repeated_keys:
                stats["removed_repeated_lines"] += 1
                continue
            if any(pat.search(line) for pat in compiled_custom):
                stats["removed_custom_pattern_lines"] += 1
                continue
            kept.append(line)
            stats["kept_lines"] += 1

        # Join lines into paragraph-like text. Blank lines are intentionally not
        # reconstructed because most PDF extractors do not preserve them well.
        text = "\n".join(kept)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        cleaned_pages.append(PDFPageText(page=page.page, text=text))

    return cleaned_pages, stats


def pages_to_text(pages: Sequence[PDFPageText], include_page_markers: bool = True) -> str:
    chunks: List[str] = []
    for page in pages:
        if include_page_markers:
            chunks.append(f"<PAGE {page.page}>\n{page.text}".strip())
        else:
            chunks.append(page.text.strip())
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _regex_sentence_split(text: str) -> List[str]:
    """Vietnamese sentence splitter fallback.

    Underthesea is preferred. This fallback avoids hard dependency on it so that
    the notebook still runs in restricted environments.
    """
    text = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if not text:
        return []
    # Split after sentence-final punctuation, but keep punctuation.
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def get_vietnamese_sentence_tokenizer(prefer_underthesea: bool = True):
    if prefer_underthesea:
        try:
            from underthesea import sent_tokenize

            return sent_tokenize, "underthesea"
        except Exception:
            pass
    return _regex_sentence_split, "regex"


def segment_vietnamese_pages(
    pages: Sequence[PDFPageText],
    work_id: str,
    prefer_underthesea: bool = True,
    min_sentence_chars: int = 2,
) -> Tuple[List[SentenceRecord], str]:
    tokenizer, tokenizer_name = get_vietnamese_sentence_tokenizer(prefer_underthesea=prefer_underthesea)
    records: List[SentenceRecord] = []
    sent_id = 1

    for page in pages:
        # Treat each non-empty line as a paragraph candidate. This helps avoid
        # merging unrelated headings and body text into a single long segment.
        paragraphs = [p.strip() for p in re.split(r"\n+", page.text) if p.strip()]
        for para in paragraphs:
            normalized_para = re.sub(r"\s+", " ", para).strip()
            if not normalized_para:
                continue
            try:
                sentences = tokenizer(normalized_para)
            except Exception:
                sentences = _regex_sentence_split(normalized_para)
                tokenizer_name = "regex"

            for sent in sentences:
                sent = re.sub(r"\s+", " ", normalize_text(sent)).strip()
                if len(sent) < min_sentence_chars:
                    continue
                records.append(SentenceRecord(work_id=work_id, lang="viet", page=page.page, sent_id=sent_id, text=sent))
                sent_id += 1

    return records, tokenizer_name


def save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_sentence_txt(path: str | Path, records: Sequence[SentenceRecord]) -> None:
    save_text(path, "\n".join(record.text for record in records))


def save_sentence_jsonl(path: str | Path, records: Sequence[SentenceRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def save_sentence_csv(path: str | Path, records: Sequence[SentenceRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["work_id", "lang", "page", "sent_id", "text"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def save_sentence_xlsx(path: str | Path, records: Sequence[SentenceRecord]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and openpyxl to export xlsx: pip install pandas openpyxl") from exc
    df = pd.DataFrame([record.__dict__ for record in records])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def process_vietnamese_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    work_id: Optional[str] = None,
    start_page: int = 1,
    end_page: Optional[int] = None,
    prefer_underthesea: bool = True,
    custom_drop_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the full Task 2 flow for one Vietnamese PDF."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    work_id = work_id or pdf_path.stem

    raw_pages = extract_pdf_pages(pdf_path, start_page=start_page, end_page=end_page)
    cleaned_pages, clean_stats = clean_vietnamese_pages(raw_pages, custom_drop_patterns=custom_drop_patterns)
    sentences, tokenizer_name = segment_vietnamese_pages(cleaned_pages, work_id=work_id, prefer_underthesea=prefer_underthesea)

    raw_text = pages_to_text(raw_pages, include_page_markers=True)
    clean_text = pages_to_text(cleaned_pages, include_page_markers=True)

    raw_txt = output_dir / f"{work_id}_viet_raw.txt"
    clean_txt = output_dir / f"{work_id}_viet_clean.txt"
    sent_txt = output_dir / f"{work_id}_viet_sentences.txt"
    sent_jsonl = output_dir / f"{work_id}_viet_sentences.jsonl"
    sent_csv = output_dir / f"{work_id}_viet_sentences.csv"
    sent_xlsx = output_dir / f"{work_id}_viet_sentences.xlsx"
    stats_json = output_dir / f"{work_id}_viet_task2_stats.json"

    save_text(raw_txt, raw_text)
    save_text(clean_txt, clean_text)
    save_sentence_txt(sent_txt, sentences)
    save_sentence_jsonl(sent_jsonl, sentences)
    save_sentence_csv(sent_csv, sentences)
    try:
        save_sentence_xlsx(sent_xlsx, sentences)
        xlsx_path = str(sent_xlsx)
    except Exception:
        xlsx_path = None

    stats = {
        "work_id": work_id,
        "pdf_path": str(pdf_path),
        "start_page": start_page,
        "end_page": end_page,
        "raw_pages": len(raw_pages),
        "clean_stats": clean_stats,
        "tokenizer": tokenizer_name,
        "num_sentences": len(sentences),
        "outputs": {
            "raw_txt": str(raw_txt),
            "clean_txt": str(clean_txt),
            "sentences_txt": str(sent_txt),
            "sentences_jsonl": str(sent_jsonl),
            "sentences_csv": str(sent_csv),
            "sentences_xlsx": xlsx_path,
            "stats_json": str(stats_json),
        },
    }
    save_text(stats_json, json.dumps(stats, ensure_ascii=False, indent=2))
    return stats
