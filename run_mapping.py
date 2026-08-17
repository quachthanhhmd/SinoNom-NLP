import os
import re
import pandas as pd
import argparse
import hashlib
import json
from typing import List, Dict, Tuple
from nlp.aligner import EmbeddingSentenceAligner, EnsembleSentenceAligner
from nlp.alignment_core import validate_alignment_records
from utils.exporters import CorpusExporter, clean_vietnamese_text
from utils.source_boundaries import split_merged_sino_rows

PIPELINE_CACHE_VERSION = "monotonic-mn-v2"


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
    """Robustly parses Sino CSVs with unquoted commas and multiline fields using python's csv module."""
    import csv
    sep = detect_separator(file_path)
    ids = []
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep, doublequote=True, skipinitialspace=True)
        try:
            # Skip header
            header = next(reader)
        except StopIteration:
            return pd.DataFrame({"ID": [], "sentence": []})
            
        for row in reader:
            if not row:
                continue
            sent_id = row[0].strip()
            
            # Reconstruct sentence when containing unquoted commas
            if len(row) >= 2:
                ref_candidate = row[-1].strip()
                # Check if last element looks like a reference_Id array e.g. [], [1, 2]
                if (ref_candidate.startswith('[') and ref_candidate.endswith(']')) or ref_candidate == '[]' or ref_candidate.startswith('"['):
                    sentence = sep.join(row[1:-1])
                else:
                    sentence = sep.join(row[1:])
            else:
                sentence = ""
                
            # Clean up newlines and quotes within the sentence
            sentence = sentence.replace('\r', '').replace('\n', ' ').strip()
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1].strip()
            sentence = sentence.replace('""', '"')
            
            # Skip empty entries
            if not sent_id and not sentence:
                continue
                
            ids.append(sent_id)
            sentences.append(sentence)
            
    return pd.DataFrame({"ID": ids, "sentence": sentences})


def split_merged_sino_dataframe(df: pd.DataFrame, volume_codes: List[str]) -> Dict[str, pd.DataFrame]:
    """Split a multi-volume OCR file by headings in its content, never by its IDs.

    The checked source files keep Q10/Q16 prefixes after the Q11/Q17 headings,
    so their ID prefix is metadata, not evidence of the actual volume.
    """
    rows = split_merged_sino_rows(df.to_dict(orient="records"), volume_codes)
    return {volume: pd.DataFrame(items) for volume, items in rows.items()}


def alignment_fingerprint(paths: List[str], settings: Dict) -> str:
    """Hash source bytes and alignment settings so stale caches cannot be reused."""
    digest = hashlib.sha256(PIPELINE_CACHE_VERSION.encode("utf-8"))
    for path in sorted(paths, key=lambda item: os.path.basename(item).lower()):
        # Hash the logical filename and bytes, not a temporary directory name.
        digest.update(os.path.basename(path).lower().encode("utf-8"))
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()[:16]

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
    sino_source_ids = []
    sino_original_ids = []
    vol_raw_text = {}    # vol_code -> raw Han text string (for _raw.txt export)
    han_breaks = []
    
    for vol_code, sino_path in sino_files:
        if all_sino_sentences:
            han_breaks.append(len(all_sino_sentences))
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
                # A CSV row is one traceable source unit.  Whitespace is soft
                # layout information and must not manufacture new sentences.
                normalized = re.sub(r"\s+", " ", sent).strip()
                all_sino_sentences.append(normalized)
                sino_source_map.append(vol_code)
                original_id = str(row.get("ID", "")).strip()
                sino_source_ids.append(
                    f"{os.path.basename(sino_path)}:{row.name}:{original_id}"
                )
                sino_original_ids.append(original_id)
                vol_sents_raw.append(normalized)
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
    viet_source_ids = []
    viet_raw_sentences = []
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
            original_viet_id = str(row.get("ID", row.name))
            viet_source_ids.append(
                f"{os.path.basename(viet_file)}:{row.name}:{original_viet_id}"
            )
            viet_raw_sentences.append(sent)
            
    print(f"[Cleaner Log] Summary: Cleaned annotations/Han characters in {cleaned_count} sentence(s) out of {len(df_viet)} total rows.")
    
    if not viet_sentences:
        raise ValueError(f"No valid Vietnamese sentences found after cleaning in {viet_file}.")
        
    print(f"Total Han sentences to align: {len(all_sino_sentences)}")
    print(f"Total Viet sentences to align (after cleaning): {len(viet_sentences)}")
    
    # Cache setup
    from config import ENSEMBLE_CONFIG
    group_key = os.path.splitext(os.path.basename(viet_file))[0]
    cache_dir = os.path.join(exporter.output_dir, work_code, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fingerprint = alignment_fingerprint(
        [path for _, path in sino_files] + [viet_file],
        {"ensemble": ENSEMBLE_CONFIG, "volumes": [volume for volume, _ in sino_files]},
    )
    phase1_cache = os.path.join(cache_dir, f"{group_key}_{fingerprint}_phase1.json")
    phase2_cache = os.path.join(cache_dir, f"{group_key}_{fingerprint}_phase2.json")
    phase3_cache = os.path.join(cache_dir, f"{group_key}_{fingerprint}_phase3.json")
    
    raw_aligned = None
    run_phase1 = True
    run_phase2 = True
    run_phase3 = True
    
    # 3. Perform Alignment (or load from cache if available)
    if os.path.exists(phase3_cache):
        print(f"[Cache] Found Phase 3 cached/checkpoint alignment at: {phase3_cache}")
        print(f"[Cache] Loading cache to skip Phase 1, 2, and 3 computation...")
        with open(phase3_cache, "r", encoding="utf-8") as f:
            raw_aligned = json.load(f)
        run_phase1 = False
        run_phase2 = False
        run_phase3 = False
    elif os.path.exists(phase2_cache):
        print(f"[Cache] Found Phase 2 cached alignment at: {phase2_cache}")
        print(f"[Cache] Loading cache to skip Phase 1 and Phase 2 computation...")
        with open(phase2_cache, "r", encoding="utf-8") as f:
            raw_aligned = json.load(f)
        run_phase1 = False
        run_phase2 = False
    elif os.path.exists(phase1_cache):
        print(f"[Cache] Found Phase 1 cached alignment at: {phase1_cache}")
        print(f"[Cache] Loading cache to skip Phase 1 computation...")
        with open(phase1_cache, "r", encoding="utf-8") as f:
            raw_aligned = json.load(f)
        run_phase1 = False
        
    if run_phase1 or raw_aligned is None:
        # Prevent a merged transition from swallowing a verified volume edge.
        aligner._han_breaks = han_breaks
        raw_aligned = aligner.align(all_sino_sentences, viet_sentences)
        validate_alignment_records(raw_aligned, len(all_sino_sentences), len(viet_sentences))
        # Save Phase 1 cache
        with open(phase1_cache, "w", encoding="utf-8") as f:
            json.dump(raw_aligned, f, ensure_ascii=False, indent=2)
        print(f"[Cache] Saved Phase 1 alignment cache to: {phase1_cache}")
    
    # 3.5 Phase 2: Qwen LLM Verification
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
                        "han_indices": item.get("han_indices", []),
                        "viet_indices": [],
                        "confidence": 0.0,
                        "status": "unmatched",
                    })
                if item["viet_sentence"]:
                    filtered_aligned.append({
                        "han_sentence": "",
                        "viet_sentence": item["viet_sentence"],
                        "similarity_score": 0.0,
                        "han_indices": [],
                        "viet_indices": item.get("viet_indices", []),
                        "confidence": 0.0,
                        "status": "unmatched",
                    })
            else:
                filtered_aligned.append(item)
        raw_aligned = filtered_aligned
        validate_alignment_records(raw_aligned, len(all_sino_sentences), len(viet_sentences))
        
        # Save Phase 2 cache
        with open(phase2_cache, "w", encoding="utf-8") as f:
            json.dump(raw_aligned, f, ensure_ascii=False, indent=2)
        print(f"[Cache] Saved Phase 2 alignment cache to: {phase2_cache}")
        
        # Giải phóng VRAM của Qwen sau khi hoàn tất Phase 2 để quyển sau chạy tiếp Phase 1 không bị OOM
        print("[VRAM] Releasing Qwen verifier from GPU VRAM after Phase 2 completion...")
        verifier.free_gpu_memory()

    # 3.7 Phase 3: Qwen Local Re-Alignment of unresolved NaN clusters
    if realign_enabled and run_phase3 and isinstance(aligner, EnsembleSentenceAligner):
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

    validate_alignment_records(raw_aligned, len(all_sino_sentences), len(viet_sentences))

    # 4. Map pairs back to verified volume boundaries using source indices.
    volume_aligned_data = {} # vol_code -> list of aligned dicts

    known_record_volumes = []
    for item in raw_aligned:
        h_idxs = item.get("han_indices", [])
        if not h_idxs:
            known_record_volumes.append(None)
            continue
        item_volumes = {sino_source_map[index] for index in h_idxs}
        if len(item_volumes) != 1:
            raise ValueError(
                f"alignment span crosses verified volume boundary: {sorted(item_volumes)}"
            )
        known_record_volumes.append(next(iter(item_volumes)))

    known_positions = [
        position for position, volume in enumerate(known_record_volumes) if volume is not None
    ]
    for position, item in enumerate(raw_aligned):
        han_txt = item["han_sentence"]
        h_idxs = item.get("han_indices", [])
        if not h_idxs:
            # A one-sided Vietnamese record has no intrinsic Han volume.  Use
            # the nearest monotonic Han record (next wins a boundary tie) and
            # keep it in the unmatched queue for human review.
            nearest_position = min(
                known_positions,
                key=lambda candidate: (abs(candidate - position), candidate < position),
            )
            vol = known_record_volumes[nearest_position]
        else:
            first_idx = h_idxs[0]
            if not (0 <= first_idx < len(sino_source_map)):
                raise ValueError(
                    f"Han index {first_idx} is out of bounds in sino_source_map (length {len(sino_source_map)}). "
                    f"Offending sentence: {repr(han_txt)}"
                )
            vol = sino_source_map[first_idx]
                
        if vol not in volume_aligned_data:
            volume_aligned_data[vol] = []
            
        enriched = dict(item)
        enriched["han_source_ids"] = [sino_source_ids[index] for index in h_idxs]
        enriched["han_original_ids"] = [sino_original_ids[index] for index in h_idxs]
        enriched["viet_source_ids"] = [
            viet_source_ids[index] for index in item.get("viet_indices", [])
        ]
        enriched["viet_raw_sentences"] = [
            viet_raw_sentences[index] for index in item.get("viet_indices", [])
        ]
        enriched["volume"] = vol
        volume_aligned_data[vol].append(enriched)
        
    # 5. Export each volume to its hierarchical directory
    for vol_code, aligned_list in volume_aligned_data.items():
        # Re-number the pair_ids for this specific volume starting at 1
        formatted_aligned = []
        for idx, item in enumerate(aligned_list):
            formatted_item = dict(item)
            formatted_item.update({
                "pair_id": f"{work_code}_{vol_code}_{idx+1:06d}",
            })
            formatted_aligned.append(formatted_item)
            
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
                
                expected_volumes = ["10", "11"] if fname.startswith("q10") else ["16", "17"]
                split_frames = split_merged_sino_dataframe(df_temp, expected_volumes)

                # Temporary files live outside the source corpus and are
                # removed automatically.  The original OCR extracts are never
                # renamed or rewritten.
                import tempfile
                with tempfile.TemporaryDirectory(prefix="sinonom_volume_split_") as temp_dir:
                    temp_sino_files = []
                    for vol_num in expected_volumes:
                        temp_csv_path = os.path.join(temp_dir, f"volume_{vol_num}.csv")
                        split_frames[vol_num].to_csv(temp_csv_path, index=False)
                        temp_sino_files.append((vol_num, temp_csv_path))

                    process_alignment_group(
                        sino_files=temp_sino_files,
                        viet_file=viet_file,
                        aligner=aligner,
                        exporter=exporter,
                        work_code=args.work_code,
                        qwen_enabled=args.qwen,
                        realign_enabled=args.realign
                    )
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
