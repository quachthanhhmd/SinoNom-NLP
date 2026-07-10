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

    def export_hierarchical(self, parent_dir: str, chapter_str: str, aligned_data: List[Dict[str, str]]):
        """
        Exports parallel TSV and Excel files into a hierarchical directory structure:
        output_dir / parent_dir / parent_dir_chapter / parent_dir_chapter_parallel.tsv
        Example: output / HVB_001 / HVB_001_01 / HVB_001_01_parallel.tsv
        """
        target_dir = os.path.join(self.output_dir, parent_dir, f"{parent_dir}_{chapter_str}")
        os.makedirs(target_dir, exist_ok=True)
        
        tsv_path = os.path.join(target_dir, f"{parent_dir}_{chapter_str}_parallel.tsv")
        xlsx_path = os.path.join(target_dir, f"{parent_dir}_{chapter_str}_parallel.xlsx")
        
        df = pd.DataFrame(aligned_data)
        if not df.empty:
            df["viet_sentence"] = df["viet_sentence"].apply(clean_vietnamese_text)
            df = df[["pair_id", "han_sentence", "viet_sentence", "similarity_score"]]
        else:
            df = pd.DataFrame(columns=["pair_id", "han_sentence", "viet_sentence", "similarity_score"])
            
        df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
        df.to_excel(xlsx_path, index=False)
        print(f"Exported hierarchical outputs to: {target_dir}")

