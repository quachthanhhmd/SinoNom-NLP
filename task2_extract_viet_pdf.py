#!/usr/bin/env python3
"""CLI for Task 2: extract, clean, and split Vietnamese PDF text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlp.vietnamese_pdf_processor import process_vietnamese_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Vietnamese PDF extraction, cleaning, and sentence segmentation")
    parser.add_argument("--pdf", required=True, help="Path to Vietnamese PDF file")
    parser.add_argument("--output_dir", default="output_task2", help="Directory for Task 2 outputs")
    parser.add_argument("--work_id", default=None, help="Work id. Defaults to PDF stem")
    parser.add_argument("--start_page", type=int, default=1, help="1-based first page to extract")
    parser.add_argument("--end_page", type=int, default=None, help="1-based last page to extract")
    parser.add_argument("--no_underthesea", action="store_true", help="Use regex sentence splitting instead of underthesea")
    parser.add_argument(
        "--drop_pattern",
        action="append",
        default=[],
        help="Regex line pattern to drop during cleaning. Can be passed multiple times.",
    )
    args = parser.parse_args()

    stats = process_vietnamese_pdf(
        pdf_path=Path(args.pdf),
        output_dir=Path(args.output_dir),
        work_id=args.work_id,
        start_page=args.start_page,
        end_page=args.end_page,
        prefer_underthesea=not args.no_underthesea,
        custom_drop_patterns=args.drop_pattern,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
