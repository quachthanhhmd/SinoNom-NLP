import os
import pandas as pd
from typing import List, Dict

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
            # Ensure column order matches requirements: [pair_id]\t[han_sentence]\t[viet_sentence]
            df = df[["pair_id", "han_sentence", "viet_sentence"]]
        df.to_csv(path, sep="\t", index=False)
        print(f"Exported parallel TSV to {path}")

    def export_parallel_excel(self, work_id: str, aligned_data: List[Dict[str, str]]):
        path = os.path.join(self.output_dir, f"{work_id}_parallel.xlsx")
        df = pd.DataFrame(aligned_data)
        if not df.empty:
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
            df = df[["pair_id", "han_sentence", "viet_sentence", "similarity_score"]]
        else:
            df = pd.DataFrame(columns=["pair_id", "han_sentence", "viet_sentence", "similarity_score"])
            
        df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
        df.to_excel(xlsx_path, index=False)
        print(f"Exported hierarchical outputs to: {target_dir}")

