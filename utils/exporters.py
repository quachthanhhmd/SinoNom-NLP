import os
import re
import pandas as pd
from typing import List, Dict

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

    def export_hierarchical(self, parent_dir: str, chapter_str: str, aligned_data: List[Dict[str, str]],
                            han_raw_text: str = None):
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
            df_full = df_full[["pair_id", "han_sentence", "viet_sentence", "similarity_score"]]
        else:
            df_full = pd.DataFrame(columns=["pair_id", "han_sentence", "viet_sentence", "similarity_score"])

        # ── Phiên bản đầy đủ nội bộ (4 cột, giữ nguyên các dòng NaN) ──────────
        full_tsv_path = os.path.join(target_dir, f"{file_prefix}_parallel_full.tsv")
        df_full.to_csv(full_tsv_path, sep="\t", index=False, encoding="utf-8")

        # ── Phiên bản chuẩn yêu cầu (3 cột, lọc bỏ dòng NaN) ─────────────────
        # Chỉ giữ các dòng có cả Hán lẫn Việt hợp lệ
        def _is_valid(val):
            if val is None:
                return False
            s = str(val).strip()
            return bool(s) and s.lower() != "nan"

        mask = df_full["han_sentence"].apply(_is_valid) & df_full["viet_sentence"].apply(_is_valid)
        df_clean = df_full[mask][["pair_id", "han_sentence", "viet_sentence"]].reset_index(drop=True)

        clean_tsv_path = os.path.join(target_dir, f"{file_prefix}_parallel.tsv")
        clean_xlsx_path = os.path.join(target_dir, f"{file_prefix}_parallel.xlsx")
        df_clean.to_csv(clean_tsv_path, sep="\t", index=False, encoding="utf-8")
        df_clean.to_excel(clean_xlsx_path, index=False)

        # ── Raw Hán text (nếu được cung cấp) ─────────────────────────────────
        if han_raw_text is not None:
            raw_path = os.path.join(target_dir, f"{file_prefix}_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(han_raw_text)
            print(f"  Exported raw text: {raw_path}")

        n_full = len(df_full)
        n_clean = len(df_clean)
        print(f"Exported hierarchical outputs to: {target_dir}")
        print(f"  _parallel.tsv (chuẩn yêu cầu): {n_clean} cặp sạch / {n_full} tổng dòng")
        print(f"  _parallel_full.tsv (nội bộ):    {n_full} dòng (bao gồm NaN)")

