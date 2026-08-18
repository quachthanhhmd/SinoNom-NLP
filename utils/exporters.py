import os
import re
import json
import pandas as pd
from typing import List, Dict, Optional

def clean_vietnamese_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    # 1. Remove year annotations like (1805), (1802-1820), (năm 1820)
    text = re.sub(r'\(\s*(?:năm\s+)?\d{4}(?:\s*-\s*\d{4})?\s*\)', '', text)
    
    # 2. Remove footnote numbers in brackets like (1), (2)
    text = re.sub(r'\(\s*\d+\s*\)', '', text)
    
    # 3. Remove Chinese (Han/Nom) characters along with any trailing punctuation/space
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+(?:\s*[,;，；、]\s*)?', '', text)
    
    # 4. Remove short parenthetical expressions (<= 15 characters)
    # This catches (bạt xanh), (bạt vàng), (nay triệt), (quan tài), etc.
    def replace_short_parenthesis(match):
        content = match.group(1).strip()
        if len(content) <= 15:
            return ""
        return match.group(0) # Keep if longer
        
    text = re.sub(r'\(([^)]+)\)', replace_short_parenthesis, text)
    
    # 5. Clean up whitespace and punctuation issues caused by removals
    # Remove consecutive commas/spaces/dots
    text = re.sub(r'\s+', ' ', text) # normalize spaces
    text = re.sub(r'\s*,\s*,\s*', ', ', text) # duplicate commas
    text = re.sub(r'\s*;\s*;+', ';', text) # duplicate semicolons
    text = re.sub(r'\s*,\s*\.', '.', text) # comma before dot
    
    # Remove spacing around punctuation
    text = re.sub(r'\s+([,.?;:])', r'\1', text)
    # Fix spaces after punctuation if missing
    text = re.sub(r'([,.?;:])(?=[A-Za-zĂăÂâĐđÊêÔôƠơƯư])', r'\1 ', text)
    
    # Clean up double punctuation left over
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\s*,\s*$', '', text.strip()) # trailing commas
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class CorpusExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_raw(self, work_id: str, text: str):
        path = os.path.join(self.output_dir, f"{work_id}_raw.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Exported raw text to {path}")

    def export_parallel_tsv(self, work_id: str, aligned_data: List[Dict[str, str]]):
        path = os.path.join(self.output_dir, f"{work_id}_parallel.tsv")
        df = pd.DataFrame(aligned_data)
        if not df.empty:
            df["viet_sentence"] = df["viet_sentence"].apply(clean_vietnamese_text)
            # Ensure column order matches requirements: [pair_id]\t[han_sentence]\t[viet_sentence]
            df = df[["pair_id", "han_sentence", "viet_sentence"]]
        df.to_csv(path, sep="\t", index=False)
        print(f"Exported parallel TSV to {path}")

    def export_parallel_excel(self, work_id: str, aligned_data: List[Dict[str, str]]):
        path = os.path.join(self.output_dir, f"{work_id}_parallel.xlsx")
        df = pd.DataFrame(aligned_data)
        if not df.empty:
            df["viet_sentence"] = df["viet_sentence"].apply(clean_vietnamese_text)
            df = df[["pair_id", "han_sentence", "viet_sentence", "similarity_score"]]
        df.to_excel(path, index=False)
        print(f"Exported parallel Excel to {path}")

    def export_hierarchical(
        self,
        parent_dir: str,
        chapter_str: str,
        aligned_data: List[Dict[str, str]],
        han_raw_text: str = None,
        rejected_data: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Exports two versions of parallel files into a hierarchical directory structure:

        output_dir / parent_dir / parent_dir_chapter /
            ├── parent_dir_chapter_parallel.tsv       ← Chuẩn yêu cầu: 3 cột, đã lọc NaN
            ├── parent_dir_chapter_parallel.xlsx      ← Chuẩn yêu cầu: 3 cột, đã lọc NaN
            ├── parent_dir_chapter_parallel_full.tsv  ← Nội bộ: 4 cột, giữ nguyên NaN
            └── parent_dir_chapter_raw.txt            ← Văn bản Hán thô (nếu có)

        Example:
            output / HVB_001 / HVB_001_01 / HVB_001_01_parallel.tsv
        """
        target_dir = os.path.join(self.output_dir, parent_dir, f"{parent_dir}_{chapter_str}")
        os.makedirs(target_dir, exist_ok=True)

        file_prefix = f"{parent_dir}_{chapter_str}"

        df_full = pd.DataFrame(aligned_data)
        if not df_full.empty:
            df_full["viet_sentence"] = df_full["viet_sentence"].apply(clean_vietnamese_text)
        else:
            df_full = pd.DataFrame(columns=[
                "pair_id", "han_sentence", "viet_sentence", "similarity_score",
                "confidence", "status", "han_indices", "viet_indices",
                "han_source_ids", "viet_source_ids", "volume",
            ])

        def _is_valid(val):
            if val is None:
                return False
            s = str(val).strip()
            return bool(s) and s.lower() != "nan"

        both_sides = (
            df_full["han_sentence"].apply(_is_valid)
            & df_full["viet_sentence"].apply(_is_valid)
        )
        if "status" not in df_full.columns:
            df_full["status"] = "unmatched"
            df_full.loc[both_sides, "status"] = "review"

        # ── Phiên bản đầy đủ nội bộ (4 cột, giữ nguyên các dòng NaN) ──────────
        full_tsv_path = os.path.join(target_dir, f"{file_prefix}_parallel_full.tsv")
        df_full.to_csv(full_tsv_path, sep="\t", index=False, encoding="utf-8")

        # Similarity/status alone is never sufficient. Only a bead explicitly
        # classified as exact by the completeness verifier is training-ready.
        if "completeness_label" not in df_full.columns:
            df_full["completeness_label"] = "unverified"
        accepted_mask = both_sides & (df_full["completeness_label"] == "exact")
        review_mask = both_sides & ~accepted_mask
        unmatched_mask = ~both_sides
        df_accepted = df_full[accepted_mask].reset_index(drop=True)
        df_review = df_full[review_mask].reset_index(drop=True)
        df_unmatched = df_full[unmatched_mask].reset_index(drop=True)
        df_clean = df_accepted[["pair_id", "han_sentence", "viet_sentence"]].copy()

        clean_tsv_path = os.path.join(target_dir, f"{file_prefix}_parallel.tsv")
        clean_xlsx_path = os.path.join(target_dir, f"{file_prefix}_parallel.xlsx")
        df_clean.to_csv(clean_tsv_path, sep="\t", index=False, encoding="utf-8")
        df_clean.to_excel(clean_xlsx_path, index=False)

        df_accepted.to_csv(
            os.path.join(target_dir, f"{file_prefix}_accepted.tsv"),
            sep="\t", index=False, encoding="utf-8",
        )
        df_accepted.to_csv(
            os.path.join(target_dir, f"{file_prefix}_exact_accepted.tsv"),
            sep="\t", index=False, encoding="utf-8",
        )
        df_review.to_csv(
            os.path.join(target_dir, f"{file_prefix}_review.tsv"),
            sep="\t", index=False, encoding="utf-8",
        )
        df_unmatched.to_csv(
            os.path.join(target_dir, f"{file_prefix}_unmatched.tsv"),
            sep="\t", index=False, encoding="utf-8",
        )

        # Keep every rejected candidate bead for debugging and targeted repair.
        # These are attempts, not training rows; a later exact repair can still
        # coexist with the earlier addition/omission diagnostic.
        df_rejected = pd.DataFrame(rejected_data or [])
        if not df_rejected.empty:
            df_rejected["viet_sentence"] = df_rejected["viet_sentence"].apply(
                clean_vietnamese_text
            )
        rejection_columns = list(df_rejected.columns) or [
            "pair_id", "han_sentence", "viet_sentence", "completeness_label",
            "extra_side", "missing_side", "verification_confidence",
            "verification_reason", "repair_round", "han_indices", "viet_indices",
        ]
        for label in ("addition", "omission", "mismatch"):
            if df_rejected.empty or "completeness_label" not in df_rejected.columns:
                partition = pd.DataFrame(columns=rejection_columns)
            else:
                partition = df_rejected[
                    df_rejected["completeness_label"] == label
                ].reset_index(drop=True)
            partition.to_csv(
                os.path.join(target_dir, f"{file_prefix}_{label}.tsv"),
                sep="\t", index=False, encoding="utf-8",
            )

        # Deterministic teacher-style sample. The blank audit columns make it
        # possible to perform an independent check instead of trusting the same
        # model that selected the exact data.
        sample_size = min(30, len(df_accepted))
        if sample_size:
            sample = df_accepted.sample(n=sample_size, random_state=42).copy()
        else:
            sample = df_accepted.copy()
        sample["audit_label"] = ""
        sample["auditor_notes"] = ""
        sample.to_excel(
            os.path.join(target_dir, f"{file_prefix}_evaluation_sample.xlsx"),
            index=False,
        )

        rejection_counts = (
            df_rejected.get("completeness_label", pd.Series(dtype=str))
            .value_counts()
            .to_dict()
        )
        report = {
            "volume": str(chapter_str),
            "exact_accepted": int(len(df_accepted)),
            "review": int(len(df_review)),
            "final_unmatched_rows": int(len(df_unmatched)),
            "rejected_candidate_attempts": {
                label: int(rejection_counts.get(label, 0))
                for label in ("addition", "omission", "mismatch")
            },
            "independent_sample_size": int(sample_size),
            "teacher_rule": "fail when independently audited errors exceed one third",
            "independent_result": "pending_manual_or_second_model_audit",
        }
        with open(
            os.path.join(target_dir, f"{file_prefix}_evaluation_report.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

        # ── Raw Hán text (nếu được cung cấp) ─────────────────────────────────
        if han_raw_text is not None:
            raw_path = os.path.join(target_dir, f"{file_prefix}_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(han_raw_text)
            print(f"  Exported raw text: {raw_path}")

        n_full = len(df_full)
        n_clean = len(df_clean)
        print(f"Exported hierarchical outputs to: {target_dir}")
        print(f"  _parallel.tsv / _exact_accepted.tsv: {n_clean} cặp exact / {n_full} tổng dòng")
        print(f"  _review.tsv:                    {len(df_review)} cặp cần duyệt")
        print(f"  _unmatched.tsv:                 {len(df_unmatched)} dòng một phía")
        print(f"  _addition/_omission/_mismatch:  {len(df_rejected)} lần thử bị từ chối")
        print(f"  _parallel_full.tsv (nội bộ):    {n_full} dòng (bao gồm NaN)")

