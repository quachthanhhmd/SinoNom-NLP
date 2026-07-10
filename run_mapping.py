import os
import re
import pandas as pd
import argparse
from typing import List, Dict, Tuple
from nlp.aligner import EmbeddingSentenceAligner, EnsembleSentenceAligner
from utils.exporters import CorpusExporter, clean_vietnamese_text


def detect_separator(file_path: str) -> str:
    """Detects if a CSV file uses ',' or ';' as a separator."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if ';' in first_line:
                return ';'
    except Exception:
        pass
    return ','

def load_sino_csv(file_path: str) -> pd.DataFrame:
    """Robustly parses Sino CSVs with unquoted commas in sentence values."""
    sep = detect_separator(file_path)
    ids = []
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep, 1)
            if len(parts) < 2:
                raise ValueError(f"Malformed line in Sino CSV '{file_path}': expected at least 2 fields separated by '{sep}', got: {repr(line)}")
            sent_id = parts[0].strip()
            rest = parts[1].strip()
            
            # Find the start of reference_Id
            match_idx = -1
            for pattern in [sep + '"[', sep + '[', sep + '[]', sep + '"[]"']:
                idx = rest.rfind(pattern)
                if idx > match_idx:
                    match_idx = idx
            if match_idx != -1:
                sentence = rest[:match_idx]
            else:
                idx = rest.rfind(sep)
                if idx != -1:
                    sentence = rest[:idx]
                else:
                    sentence = rest
            
            sentence = sentence.strip()
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1].strip()
            sentence = sentence.replace('""', '"')
            ids.append(sent_id)
            sentences.append(sentence)
    return pd.DataFrame({"ID": ids, "sentence": sentences})

def clean_vietnamese_sentence(sentence: str) -> bool:
    """
    Returns True if the sentence is a valid content sentence, 
    False if it looks like a heading/metadata (e.g., 'QUYỂN II', 'PHỦ THỪA THIÊN').
    """
    if not isinstance(sentence, str):
        return False
    sentence = sentence.strip()
    if not sentence:
        return False
    
    # 1. Skip extremely short strings (usually page numbers or table junk)
    if len(sentence) < 6:
        return False
        
    # 2. Skip typical uppercase metadata titles / headings (length constraint to avoid skipping short real sentences)
    # Check if mostly uppercase and short
    words = sentence.split()
    uppercase_words = [w for w in words if w.isupper() or not w.isalpha()]
    if len(uppercase_words) / len(words) > 0.8 and len(sentence) < 45:
        return False
        
    # 3. Check for typical structural headings in Sino-Nom texts
    heading_keywords = [
        r"^quyển\s+(?:[ivxlcdm]+|\d+)",
        r"^quyển\s+thứ\s+[a-ăâb-đe-êg-hi-k-l-m-n-o-ô-ơp-qr-s-t-u-ưv-xy]+",
        r"^tỉnh\s+[a-ăâb-đe-êg-hi-k-l-m-n-o-ô-ơp-qr-s-t-u-ưv-xy\-\s]+",
        r"^phủ\s+[a-ăâb-đe-êg-hi-k-l-m-n-o-ô-ơp-qr-s-t-u-ưv-xy\-\s]+",
        r"^đại\s*-\s*nam\s+nhất\s*-\s*thống\s*-\s*chí",
        r"^đại\s+nam\s+nhất\s+thống\s+chí",
        r"^dựng\s+đặt\s+và\s+diên\s+cách",
        r"^phân\s+dã",
        r"^khí\s+hậu",
        r"^thành\s*-\s*trì",
        r"^tử\s*-\s*chí",
        r"^sông\s+núi",
        r"^danh\s+lam",
        r"^cổ\s+tự",
        r"^sản\s+vật",
        r"^nhân\s+vật"
    ]
    
    sentence_lower = sentence.lower()
    for kw in heading_keywords:
        if re.search(kw, sentence_lower):
            return False
            
    return True

def process_alignment_group(
    sino_files: List[Tuple[str, str]], # list of (volume_code_str, filepath) e.g., [("02", "q2_sentences.csv")]
    viet_file: str,
    aligner: EnsembleSentenceAligner,
    exporter: CorpusExporter,
    work_code: str,
    qwen_enabled: bool = False,
    realign_enabled: bool = False
):
    vols_str = ", ".join([f"Quyển {f[0]} ({os.path.basename(f[1])})" for f in sino_files])
    print(f"\n=========================================================================")
    print(f">>> BẮT ĐẦU XỬ LÝ NHÓM FILE:")
    print(f"    - Hán Nôm: {vols_str}")
    print(f"    - Tiếng Việt: {os.path.basename(viet_file)}")
    print(f"=========================================================================")
    
    # 1. Read Sino sentences and track their source volumes
    all_sino_sentences = []
    sino_source_map = [] # tracks which index in all_sino_sentences belongs to which volume code
    vol_raw_text = {}    # vol_code -> raw Han text string (for _raw.txt export)
    
    for vol_code, sino_path in sino_files:
        if not os.path.exists(sino_path):
            raise FileNotFoundError(f"Sino file not found: {sino_path}")
        df_sino = load_sino_csv(sino_path)
        # Ensure column 'sentence' exists
        if 'sentence' not in df_sino.columns:
            raise KeyError(f"Column 'sentence' missing in {sino_path}")
            
        # Special filter for Volume 1: skip preface/tấu biểu before '大南一統志卷之一'
        is_quyen1 = (str(vol_code) == "01" or str(vol_code) == "1")
        skip_preface = is_quyen1
        
        vol_sents_raw = []
        for _, row in df_sino.iterrows():
            sent = str(row['sentence']).strip()
            if sent:
                if skip_preface:
                    if "大南一統志卷之一" in sent:
                        skip_preface = False
                    else:
                        continue
                # Split Han sentences by whitespace to break down merged lines
                parts = [p.strip() for p in re.split(r'\s+', sent) if p.strip()]
                for p in parts:
                    all_sino_sentences.append(p)
                    sino_source_map.append(vol_code)
                    vol_sents_raw.append(p)
        vol_raw_text[vol_code] = "\n".join(vol_sents_raw)
                
    if not all_sino_sentences:
        raise ValueError(f"No Sino sentences found for work {work_code}.")
        
    # 2. Read and filter Vietnamese sentences
    if not os.path.exists(viet_file):
        raise FileNotFoundError(f"Viet file not found: {viet_file}")
        
    df_viet = pd.read_csv(viet_file)
    if 'sentence' not in df_viet.columns:
        raise KeyError(f"Column 'sentence' missing in {viet_file}")
        
    viet_sentences = []
    cleaned_count = 0
    printed_examples = 0
    
    for _, row in df_viet.iterrows():
        sent = str(row['sentence']).strip()
        cleaned_sent = clean_vietnamese_text(sent)
        
        # Log modifications for the user to verify
        if cleaned_sent != sent:
            cleaned_count += 1
            if printed_examples < 5:
                print(f"[Cleaner Log] Modified sentence:")
                print(f"  Original: {repr(sent)}")
                print(f"  Cleaned:  {repr(cleaned_sent)}")
                printed_examples += 1
                
        if clean_vietnamese_sentence(cleaned_sent):
            viet_sentences.append(cleaned_sent)
            
    print(f"[Cleaner Log] Summary: Cleaned annotations/Han characters in {cleaned_count} sentence(s) out of {len(df_viet)} total rows.")
    
    if not viet_sentences:
        raise ValueError(f"No valid Vietnamese sentences found after cleaning in {viet_file}.")
        
    print(f"Total Han sentences to align: {len(all_sino_sentences)}")
    print(f"Total Viet sentences to align (after cleaning): {len(viet_sentences)}")
    
    # Cache setup
    import json
    group_key = os.path.splitext(os.path.basename(viet_file))[0]
    cache_dir = os.path.join(exporter.output_dir, work_code, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    phase1_cache = os.path.join(cache_dir, f"{group_key}_phase1.json")
    phase2_cache = os.path.join(cache_dir, f"{group_key}_phase2.json")
    phase3_cache = os.path.join(cache_dir, f"{group_key}_phase3.json")
    
    raw_aligned = None
    run_phase1 = True
    run_phase2 = True
    
    # 3. Perform Alignment (or load from cache if available)
    if realign_enabled and os.path.exists(phase3_cache):
        print(f"[Cache] Found Phase 3 cached/checkpoint alignment at: {phase3_cache}")
        print(f"[Cache] Loading cache to resume or skip Phase 1 and 2...")
        with open(phase3_cache, "r", encoding="utf-8") as f:
            raw_aligned = json.load(f)
        run_phase1 = False
        run_phase2 = False
    elif realign_enabled and os.path.exists(phase2_cache):
        print(f"[Cache] Found Phase 2 cached alignment at: {phase2_cache}")
        print(f"[Cache] Loading cache to skip Phase 1 (SimAlign) and Phase 2 (Verification)...")
        with open(phase2_cache, "r", encoding="utf-8") as f:
            raw_aligned = json.load(f)
        run_phase1 = False
        run_phase2 = False
    elif (qwen_enabled or realign_enabled) and os.path.exists(phase1_cache):
        print(f"[Cache] Found Phase 1 cached alignment at: {phase1_cache}")
        print(f"[Cache] Loading cache to skip Phase 1 computation (SimAlign)...")
        with open(phase1_cache, "r", encoding="utf-8") as f:
            raw_aligned = json.load(f)
        run_phase1 = False
        
    if run_phase1 or raw_aligned is None:
        raw_aligned = aligner.align(all_sino_sentences, viet_sentences)
        # Save Phase 1 cache
        with open(phase1_cache, "w", encoding="utf-8") as f:
            json.dump(raw_aligned, f, ensure_ascii=False, indent=2)
        print(f"[Cache] Saved Phase 1 alignment cache to: {phase1_cache}")
    
    # 3.5 Phase 2: Qwen LLM Verification
    from config import ENSEMBLE_CONFIG
    qwen_conf = ENSEMBLE_CONFIG.get("qwen_verifier", {})
    if isinstance(aligner, EnsembleSentenceAligner) and qwen_enabled and qwen_conf.get("enabled", True) and run_phase2:
        # Giải phóng VRAM từ Phase 1 (LaBSE, BERTAlign, SimAlign) TRƯỚC khi nạp Qwen 7B
        print("[VRAM] Releasing Phase 1 scorer models from GPU before loading Qwen...")
        aligner.free_gpu_memory()

        from nlp.qwen_verifier import QwenVerifier
        verifier = QwenVerifier(qwen_conf)
        raw_aligned = verifier.verify(raw_aligned, cache_path=phase2_cache)
        
        # Split rejected pairs (verified == False) into unaligned sentences to avoid false matches
        filtered_aligned = []
        for item in raw_aligned:
            if not item.get("verified", True):
                print(f"[Qwen] Splitting incorrect match (score={item.get('qwen_score')}):")
                print(f"  Han:  {item['han_sentence']}")
                print(f"  Viet: {item['viet_sentence']}")
                if item["han_sentence"]:
                    filtered_aligned.append({
                        "han_sentence": item["han_sentence"],
                        "viet_sentence": "",
                        "similarity_score": 0.0,
                        "han_indices": item.get("han_indices", [])
                    })
                if item["viet_sentence"]:
                    filtered_aligned.append({
                        "han_sentence": "",
                        "viet_sentence": item["viet_sentence"],
                        "similarity_score": 0.0,
                        "han_indices": []
                    })
            else:
                filtered_aligned.append(item)
        raw_aligned = filtered_aligned
        
        # Save Phase 2 cache
        with open(phase2_cache, "w", encoding="utf-8") as f:
            json.dump(raw_aligned, f, ensure_ascii=False, indent=2)
        print(f"[Cache] Saved Phase 2 alignment cache to: {phase2_cache}")
        
        # Giải phóng VRAM của Qwen sau khi hoàn tất Phase 2 để quyển sau chạy tiếp Phase 1 không bị OOM
        print("[VRAM] Releasing Qwen verifier from GPU VRAM after Phase 2 completion...")
        verifier.free_gpu_memory()

    # 3.7 Phase 3: Qwen Local Re-Alignment of unresolved NaN clusters
    if realign_enabled and isinstance(aligner, EnsembleSentenceAligner):
        from nlp.qwen_realigner import QwenRealigner
        # Reuse model and tokenizer if already loaded in Phase 2
        qwen_model = verifier._model if 'verifier' in locals() else None
        qwen_tokenizer = verifier._tokenizer if 'verifier' in locals() else None
        
        realigner = QwenRealigner(
            config=ENSEMBLE_CONFIG.get("qwen_verifier", {}),
            model=qwen_model,
            tokenizer=qwen_tokenizer
        )
        raw_aligned = realigner.realign(raw_aligned, cache_path=phase3_cache)

    # 4. Map the aligned pairs back to their respective volumes
    # Backtrack aligned sentences using the original indices to figure out the volume
    # raw_aligned is a list of {'pair_id': ..., 'han_sentence': ..., 'viet_sentence': ...}
    # However, since the aligner might have concatenated m-n sentences, we should be careful.
    # To map accurately, let's keep track of Han sentence indices.
    
    # Let's recreate alignment with indices mapping by running a quick match.
    # This allows us to divide the aligned results back into their original volumes.
    volume_aligned_data = {} # vol_code -> list of aligned dicts
    
    # First, let's build a lookup dictionary for all_sino_sentences to find their volume code.
    # Since we mapped them sequentially, we can track them.
    # We can reconstruct the alignment with index markers.
    
    # We run the aligner again but we extract the index mapping by matching the text
    # Or more robustly, we can modify the aligner to return index mappings.
    # To keep code clean, let's do text matching. Since Hán sentences are mostly unique in a book, 
    # we can search for the first Hán sentence in the aligned block to identify its volume.
    
    # Reconstruct the volume mapping using explicit Han index tracking (robust to out-of-order outputs)
    current_vol = sino_source_map[0]
    
    for item in raw_aligned:
        han_txt = item["han_sentence"]
        h_idxs = item.get("han_indices", [])
        
        if not han_txt or not h_idxs:
            # If Hán is empty (Viet-only sentence), we assign it to the last active volume
            vol = current_vol
        else:
            first_idx = h_idxs[0]
            if not (0 <= first_idx < len(sino_source_map)):
                raise ValueError(
                    f"Han index {first_idx} is out of bounds in sino_source_map (length {len(sino_source_map)}). "
                    f"Offending sentence: {repr(han_txt)}"
                )
            vol = sino_source_map[first_idx]
            current_vol = vol
                
        if vol not in volume_aligned_data:
            volume_aligned_data[vol] = []
            
        volume_aligned_data[vol].append({
            "han_sentence": item["han_sentence"],
            "viet_sentence": item["viet_sentence"],
            "similarity_score": item.get("similarity_score", 0.0)
        })
        
    # 5. Export each volume to its hierarchical directory
    for vol_code, aligned_list in volume_aligned_data.items():
        # Re-number the pair_ids for this specific volume starting at 1
        formatted_aligned = []
        for idx, item in enumerate(aligned_list):
            formatted_aligned.append({
                "pair_id": f"{work_code}_{vol_code}_{idx+1:06d}",
                "han_sentence": item["han_sentence"],
                "viet_sentence": item["viet_sentence"],
                "similarity_score": item.get("similarity_score", 0.0)
            })
            
        print(f"Exporting Volume {vol_code} with {len(formatted_aligned)} aligned pairs...")
        exporter.export_hierarchical(
            parent_dir=work_code,
            chapter_str=vol_code,
            aligned_data=formatted_aligned,
            han_raw_text=vol_raw_text.get(vol_code)
        )
        
        # In báo cáo sơ bộ cho từng volume
        total_pairs = len(formatted_aligned)
        valid_matches = sum(1 for item in formatted_aligned if item["han_sentence"] and item["viet_sentence"])
        nan_han = sum(1 for item in formatted_aligned if not item["han_sentence"] and item["viet_sentence"])
        nan_viet = sum(1 for item in formatted_aligned if item["han_sentence"] and not item["viet_sentence"])
        match_rate = (valid_matches / total_pairs * 100) if total_pairs > 0 else 0
        print(f"=======================================================")
        print(f"[Báo cáo sơ bộ Volume {vol_code}]:")
        print(f"  - Tổng số cặp xuất bản: {total_pairs}")
        print(f"  - Khớp song song (Hán - Việt): {valid_matches} ({match_rate:.2f}%)")
        print(f"  - Khuyết chữ Hán (Chỉ có Việt): {nan_han}")
        print(f"  - Khuyết tiếng Việt (Chỉ có Hán): {nan_viet}")
        print(f"=======================================================")
def main():
    parser = argparse.ArgumentParser(description="Hán-Việt Sentence Alignment Orchestrator")
    parser.add_argument("--sino_dir", type=str, default="dataset/MAPPING/sino_extract", help="Sino extract directory")
    parser.add_argument("--viet_dir", type=str, default="dataset/MAPPING/vietnam_extract/csv", help="Viet extract CSV directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--work_code", type=str, default="HVB_001", help="Parent work code (e.g., HVB_001)")
    parser.add_argument("--model", type=str, default="sentence-transformers/LaBSE", help="Multilingual embedding model to use")
    parser.add_argument("--device", type=str, default=None, help="Device to run embedding on (cpu/cuda)")
    parser.add_argument("--dev", action="store_true", help="Run in dev mode (only align the first group/Volume 1 for testing)")
    parser.add_argument("--aligner", type=str, default="ensemble", choices=["embedding", "ensemble"], help="Aligner implementation to use")
    parser.add_argument("--qwen", action="store_true", default=False, help="Run Qwen LLM verification filter (Phase 2)")
    parser.add_argument("--realign", action="store_true", default=False, help="Run Qwen Phase 3 local re-alignment of NaN clusters")
    
    args = parser.parse_args()
    
    # 1. Initialize Aligner and Exporter
    if args.aligner == "ensemble":
        aligner = EnsembleSentenceAligner(device=args.device)
    else:
        aligner = EmbeddingSentenceAligner(model_name=args.model, device=args.device)
    exporter = CorpusExporter(output_dir=args.output_dir)

    
    # 2. Define the Mapping Groups based on the reviewed correspondence table
    # Each group: (List of (volume_code, sino_csv_filename), viet_csv_filename)
    mapping_groups = [
        # Quyển 1
        ([("01", "q1_sentences.csv")], "q01.csv"),
        
        # Quyển 2, 3, 4 -> mapped to the combined q2_3_4 file
        ([
            ("02", "q2_sentences.csv"),
            ("03", "q3_sentences.csv"),
            ("04", "q4_sentences.csv")
         ], "q2_3_4.csv"),
        
        # Quyển 5
        ([("05", "q5_sentences.csv")], "q05.csv"),
        
        # Quyển 6 -> mapped to q6.csv
        ([("06", "q6_sentences.csv")], "q6.csv"),
        
        # Quyển 7, 8
        ([
            ("07", "q7_sentences.csv"),
            ("08", "q8_sentences.csv")
         ], "q07_08.csv"),
         
        # Quyển 9
        ([("09", "q9_sentences.csv")], "q09.csv"),
        
        # Quyển 10, 11 -> Hán has q10_11_sentences.csv. We split them into "10" and "11" outputs dynamically
        ([
            ("10", "q10_11_sentences.csv") # They are stored in one file, we mark them as "10" temporarily.
                                            # We will handle sub-volume naming inside the sino file.
         ], "q10_11.csv"),
         
        # Quyển 12
        ([("12", "q12_sentences.csv")], "q12.csv"),
        
        # Quyển 13
        ([("13", "q13_sentences.csv")], "q13.csv"),
        
        # Quyển 14, 15
        ([
            ("14", "q14_sentences.csv"),
            ("15", "q15_sentences.csv")
         ], "q14_15.csv"),
         
        # Quyển 16, 17
        ([
            ("16", "q16_17_sentences.csv")
         ], "q16_17.csv")
    ]
    
    # 3. Process each group
    if args.dev:
        print("[Dev Mode] Running only the first group (Volume 1) for quick verification...")
        mapping_groups = mapping_groups[:1]
        
    for sino_info, viet_filename in mapping_groups:
        # Construct full paths
        sino_files = []
        for vol_code, fname in sino_info:
            sino_files.append((vol_code, os.path.join(args.sino_dir, fname)))
            
        viet_file = os.path.join(args.viet_dir, viet_filename)
        
        # Special logic for q10_11 and q16_17 since the sino file contains multiple volumes.
        # We want to dynamically split the output volume code based on sentence IDs if possible.
        # e.g., IDs in q10_11_sentences.csv start with Q10_... and Q11_...
        # Let's inspect the ID column in the sino file to split the volume codes accurately.
        if len(sino_info) == 1 and sino_info[0][1] in ["q10_11_sentences.csv", "q16_17_sentences.csv"]:
            fname = sino_info[0][1]
            sino_path = os.path.join(args.sino_dir, fname)
            
            if os.path.exists(sino_path):
                print(f"\n--- Special handling for merged volume Hán file: {fname} ---")
                df_temp = load_sino_csv(sino_path)
                
                # Check IDs to split
                # e.g., ID format: Q10_2_001 -> volume "10", Q11_3_001 -> volume "11"
                # e.g., ID format: Q16_... -> volume "16", Q17_... -> volume "17"
                # We can group sentences by their volume prefix from the ID column.
                vol_sentences = {} # vol_code -> list of dicts
                
                for _, row in df_temp.iterrows():
                    sent_id = str(row['ID'])
                    sentence = str(row['sentence']).strip()
                    if not sentence:
                        continue
                    # extract volume number from ID like "Q10_2_001" -> "10"
                    match = re.match(r"[qQ](\d+)_", sent_id)
                    if match:
                        vol_num = f"{int(match.group(1)):02d}"
                    else:
                        raise ValueError(f"Unexpected ID format in merged volume file {fname}: {repr(sent_id)}")
                        
                    if vol_num not in vol_sentences:
                        vol_sentences[vol_num] = []
                    vol_sentences[vol_num].append({
                        "ID": sent_id,
                        "sentence": sentence,
                        "reference_Id": "[]"
                    })
                
                # Now we write temporary split CSV files so we can reuse the process_alignment_group logic!
                temp_sino_files = []
                for vol_num, sents_dicts in vol_sentences.items():
                    temp_csv_path = os.path.join(args.sino_dir, f"temp_split_{vol_num}.csv")
                    pd.DataFrame(sents_dicts).to_csv(temp_csv_path, index=False)
                    temp_sino_files.append((vol_num, temp_csv_path))
                
                # Sort temp files so volume "10" comes before "11", and "16" before "17"
                temp_sino_files.sort(key=lambda x: x[0])
                
                # Run alignment
                process_alignment_group(
                    sino_files=temp_sino_files,
                    viet_file=viet_file,
                    aligner=aligner,
                    exporter=exporter,
                    work_code=args.work_code,
                    qwen_enabled=args.qwen,
                    realign_enabled=args.realign
                )
                
                # Clean up temporary split files
                for _, temp_path in temp_sino_files:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                print(f"[Error] Merged sino file not found: {sino_path}")
        else:
            # Normal group processing
            process_alignment_group(
                sino_files=sino_files,
                viet_file=viet_file,
                aligner=aligner,
                exporter=exporter,
                work_code=args.work_code,
                qwen_enabled=args.qwen,
                realign_enabled=args.realign
            )

    print("\nMapping phase completed successfully!")

if __name__ == "__main__":
    main()
