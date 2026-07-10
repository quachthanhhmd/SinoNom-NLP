"""
Qwen Re-Aligner — Phase 3: Local Re-Alignment of Unresolved NaN Clusters.

After Phase 2 (QwenVerifier), some Han and Viet sentences remain unmatched (similarity_score=0,
han_sentence=NaN or viet_sentence=NaN). These typically come from 2 situations:
  1. A long cluster of Han sentences was rejected by Qwen because the embedding
     couldn't correctly fuse them with the right Viet sentences.
  2. A section where the translation style differs significantly between the two texts.

This module:
  1. Scans the aligned pairs list for consecutive NaN clusters.
  2. Groups adjacent unmatched Han sentences + unmatched Viet sentences into "unresolved blocks".
  3. For each block, sends the Han and Viet sentences to Qwen with a JSON alignment prompt.
  4. Parses Qwen's output and replaces the NaN entries with corrected alignments.
  5. Falls back gracefully (keeps NaN as-is) if Qwen returns unparseable output.

Runs offline on Kaggle T4 with 4-bit quantization. Reuses the already-loaded model if
QwenVerifier is provided.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from config import ENSEMBLE_CONFIG


class QwenRealigner:
    """
    Phase 3: LLM-driven local re-alignment of unresolved NaN clusters.

    Detects clusters of consecutive unmatched Han/Viet sentences and
    asks Qwen2.5-7B to align them locally, outputting a JSON structure.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model=None,
        tokenizer=None,
    ):
        """
        Args:
            config:     qwen_verifier config dict. Falls back to ENSEMBLE_CONFIG.
            model:      Optional pre-loaded Qwen model (to avoid re-loading from QwenVerifier).
            tokenizer:  Optional pre-loaded Qwen tokenizer.
        """
        self.config = config or ENSEMBLE_CONFIG.get("qwen_verifier", {})
        self.model_name = self.config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.load_in_4bit = self.config.get("load_in_4bit", True)
        self.device_map = self.config.get("device_map", "auto")
        self.batch_size = self.config.get("batch_size", 4)

        # Allow reusing model loaded from QwenVerifier
        self._model = model
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self):
        if self._model is not None:
            return

        print(f"[QwenRealign] Loading {self.model_name} (4-bit={self.load_in_4bit})...")
        t0 = time.time()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model_kwargs = {"device_map": self.device_map}

        if self.load_in_4bit:
            try:
                import accelerate
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("[QwenRealign] Warning: bitsandbytes not installed. Loading without 4-bit.")
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        print(f"[QwenRealign] Model loaded. ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # Cluster detection
    # ------------------------------------------------------------------ #

    def _detect_nan_clusters(
        self, aligned_pairs: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, List[str], List[str]]]:
        """
        Scan aligned_pairs for consecutive blocks of unmatched sentences.

        A "NaN cluster" is a contiguous range [start_idx, end_idx) where each
        entry has either han_sentence=NaN/empty OR viet_sentence=NaN/empty
        (but not both simultaneously from different sides).

        Returns:
            List of tuples: (start_idx, end_idx, han_sentences_list, viet_sentences_list)
            where han_sentences_list and viet_sentences_list are non-empty lists of
            the unmatched sentences in that cluster.
        """
        clusters = []
        n = len(aligned_pairs)
        i = 0

        while i < n:
            pair = aligned_pairs[i]
            han = str(pair.get("han_sentence", "") or "").strip()
            viet = str(pair.get("viet_sentence", "") or "").strip()
            is_nan = (not han or han.lower() == "nan") or (not viet or viet.lower() == "nan")

            if not is_nan:
                i += 1
                continue

            # Found start of NaN cluster — scan forward to find the full extent
            cluster_start = i
            cluster_han = []
            cluster_viet = []

            while i < n:
                p = aligned_pairs[i]
                h = str(p.get("han_sentence", "") or "").strip()
                v = str(p.get("viet_sentence", "") or "").strip()
                h_is_nan = not h or h.lower() == "nan"
                v_is_nan = not v or v.lower() == "nan"

                if not h_is_nan and not v_is_nan:
                    # Found a valid pair — end of cluster
                    break

                if not h_is_nan:
                    cluster_han.append(h)
                if not v_is_nan:
                    cluster_viet.append(v)

                i += 1

            cluster_end = i

            # Only attempt re-alignment if we have BOTH sides to work with
            if cluster_han and cluster_viet:
                clusters.append((cluster_start, cluster_end, cluster_han, cluster_viet))
            # else: one-sided orphan blocks — leave as NaN (nothing to align against)

        return clusters

    # ------------------------------------------------------------------ #
    # Prompt builder
    # ------------------------------------------------------------------ #

    def _build_realign_prompt(self, han_sentences: List[str], viet_sentences: List[str]) -> str:
        han_numbered = "\n".join(f"  H{i+1}: {s}" for i, s in enumerate(han_sentences))
        viet_numbered = "\n".join(f"  V{j+1}: {s}" for j, s in enumerate(viet_sentences))

        system_content = (
            "Bạn là chuyên gia Hán Nôm và dịch thuật cổ văn Việt Nam với kinh nghiệm sâu rộng "
            "về Đại Nam Nhất Thống Chí."
        )
        user_content = f"""Dưới đây là các câu chữ Hán cổ và các câu dịch tiếng Việt tương ứng bị lệch pha (không được dóng hàng đúng chỗ).

Nhiệm vụ của bạn: Dóng hàng lại các câu Hán và Việt dưới đây một cách CHÍNH XÁC NHẤT.

QUY TẮC:
- Mỗi cặp JSON phải có "han" (chứa mã tham chiếu câu Hán, ví dụ: "H1", hoặc nhiều câu gộp như "H1+H2") và "viet" (chứa mã tham chiếu câu Việt, ví dụ: "V1", hoặc nhiều câu gộp như "V1+V2").
- BẮT BUỘC dùng mã số thứ tự H1, H2... cho Hán và V1, V2... cho Việt để điền vào trường "han" và "viet". KHÔNG ĐƯỢC tự ý viết lại toàn bộ nội dung văn bản gốc vào JSON.
- Mỗi câu CHỈ được dùng một lần.
- Nếu một câu Hán không tìm được câu Việt phù hợp, đặt "viet": null.
- Nếu một câu Việt không tìm được câu Hán phù hợp, đặt "han": null.
- Trả lời ĐÚNG định dạng JSON, không giải thích thêm.

CÁC CÂU HÁN:
{han_numbered}

CÁC CÂU VIỆT:
{viet_numbered}

Ví dụ kết quả dóng hàng mong muốn (JSON array):
[
  {{"han": "H1", "viet": "V1"}},
  {{"han": "H2+H3", "viet": "V2"}},
  {{"han": "H4", "viet": null}},
  {{"han": null, "viet": "V3"}}
]
"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------ #
    # JSON parser
    # ------------------------------------------------------------------ #

    def _parse_realign_response(
        self,
        response_text: str,
        han_sentences: List[str],
        viet_sentences: List[str],
        global_han_indices: List[int],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Parse Qwen's JSON output and resolve H/V references to actual sentence text.

        Returns None if parsing fails.
        """
        clean_text = response_text.strip()
        # Clean markdown code blocks if present
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[-1].split("```")[0].strip()

        # Find the actual JSON array start (matching '[' followed by space/braces)
        match_start = re.search(r'\[\s*\{', clean_text)
        if not match_start:
            return None
        start_idx = match_start.start()
        
        # Find matching closing bracket ']'
        end_idx = clean_text.rfind(']')
        if end_idx == -1 or end_idx < start_idx:
            return None
        
        json_str = clean_text[start_idx : end_idx + 1]

        # Basic JSON cleanup
        json_str = re.sub(r',\s*\}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)

        raw_pairs = None
        try:
            # Standard JSON load
            raw_pairs = json.loads(json_str)
        except json.JSONDecodeError:
            # Robust fallback parser for single/double quotes, formatting, and key order
            cleaned = re.sub(r"'\s*han\s*'", '"han"', json_str)
            cleaned = re.sub(r"'\s*viet\s*'", '"viet"', cleaned)
            cleaned = re.sub(r"'\s*(H\d+|V\d+|null)\s*'", lambda m: '"' + m.group(1) + '"' if m.group(1) != 'null' else 'null', cleaned)
            
            try:
                raw_pairs = json.loads(cleaned)
            except json.JSONDecodeError:
                # Direct dictionary block extractor (bulletproof regex fallback)
                dict_blocks = re.findall(r'\{[^{}]+\}', cleaned)
                if dict_blocks:
                    raw_pairs = []
                    for block in dict_blocks:
                        han_m = re.search(r'"han"\s*:\s*([^,}\s\n]+)', block)
                        if not han_m:
                            han_m = re.search(r"'han'\s*:\s*([^,}\s\n]+)", block)
                        
                        viet_m = re.search(r'"viet"\s*:\s*([^,}\s\n]+)', block)
                        if not viet_m:
                            viet_m = re.search(r"'viet'\s*:\s*([^,}\s\n]+)", block)
                            
                        if han_m or viet_m:
                            h_val = han_m.group(1).strip().strip('"').strip("'") if han_m else None
                            v_val = viet_m.group(1).strip().strip('"').strip("'") if viet_m else None
                            if h_val and h_val.lower() == 'null': h_val = None
                            if v_val and v_val.lower() == 'null': v_val = None
                            raw_pairs.append({"han": h_val, "viet": v_val})

        if not raw_pairs or not isinstance(raw_pairs, list):
            return None

        results = []
        for item in raw_pairs:
            if not isinstance(item, dict):
                continue

            # Resolve "H1", "H2"... references to actual text
            han_val = item.get("han")
            viet_val = item.get("viet")

            han_text = self._resolve_text(han_val, han_sentences, "H")
            viet_text = self._resolve_text(viet_val, viet_sentences, "V")

            # Resolve references to actual index list
            resolved_indices = []
            if han_val is not None:
                val_str = str(han_val).strip()
                refs = re.findall(r"H(\d+)", val_str, re.IGNORECASE)
                for ref in refs:
                    idx = int(ref) - 1
                    if 0 <= idx < len(global_han_indices):
                        resolved_indices.extend(global_han_indices[idx])

            if han_text or viet_text:
                results.append({
                    "han_sentence": han_text or "",
                    "viet_sentence": viet_text or "",
                    "similarity_score": 0.75 if (han_text and viet_text) else 0.0,
                    "qwen_realigned": True,
                    "han_indices": resolved_indices
                })

        return results if results else None

    def _resolve_text(
        self, val: Any, sentences: List[str], prefix: str
    ) -> Optional[str]:
        """
        Resolve val to actual sentence text.
        val can be:
          - None/null → return None
          - A string containing references like "H1" or "H1 H2" → look up by index
          - A plain sentence string → return as-is
        """
        if val is None:
            return None
        val = str(val).strip()
        if not val or val.lower() == "null":
            return None

        # Check if it looks like reference format: "H1", "H1 H2", "V1 V2 V3"
        refs = re.findall(rf"{prefix}(\d+)", val, re.IGNORECASE)
        if refs:
            parts = []
            for ref in refs:
                idx = int(ref) - 1  # 1-indexed to 0-indexed
                if 0 <= idx < len(sentences):
                    parts.append(sentences[idx])
            return " ".join(parts) if parts else None

        # Otherwise treat as plain text returned directly by Qwen
        return val

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def realign(self, aligned_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phase 3: Detect NaN clusters and re-align them using Qwen.

        Args:
            aligned_pairs: List of dicts from Phase 2 output.

        Returns:
            Updated list with NaN clusters replaced by Qwen's re-alignments.
        """
        clusters = self._detect_nan_clusters(aligned_pairs)
        if not clusters:
            print("[QwenRealign] No NaN clusters found. Phase 3 skipped.")
            return aligned_pairs

        print(f"[QwenRealign] Found {len(clusters)} NaN cluster(s) to re-align.")
        self._load_model()

        # Clear VRAM before starting realignments
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        total_fixed = 0
        # Process clusters in reverse order so index replacement doesn't shift positions
        for cluster_start, cluster_end, cluster_han, cluster_viet in reversed(clusters):
            # Extract Han indices for the cluster (preserving 1-to-1 mapping even for merged sentences)
            cluster_han_indices = []
            for item in aligned_pairs[cluster_start:cluster_end]:
                h = str(item.get("han_sentence", "") or "").strip()
                h_is_nan = not h or h.lower() == "nan"
                if not h_is_nan:
                    cluster_han_indices.append(item.get("han_indices", []))

            # Proportional slicing for large clusters to prevent VRAM OOM
            sub_clusters = []
            if len(cluster_han) > 12 or len(cluster_viet) > 12:
                n_han = len(cluster_han)
                n_viet = len(cluster_viet)
                max_len = max(n_han, n_viet)
                num_chunks = (max_len + 11) // 12
                
                h_step = max(1, int(n_han / num_chunks))
                v_step = max(1, int(n_viet / num_chunks))
                
                print(
                    f"[QwenRealign] Cluster [{cluster_start}:{cluster_end}] is large "
                    f"({n_han} Han, {n_viet} Viet). Slicing into {num_chunks} sub-cluster(s)..."
                )
                
                for chunk_no in range(num_chunks):
                    h_start = chunk_no * h_step
                    h_end = h_start + h_step if chunk_no < num_chunks - 1 else n_han
                    
                    v_start = chunk_no * v_step
                    v_end = v_start + v_step if chunk_no < num_chunks - 1 else n_viet
                    
                    sub_clusters.append((
                        cluster_han[h_start:h_end],
                        cluster_viet[v_start:v_end],
                        cluster_han_indices[h_start:h_end]
                    ))
            else:
                sub_clusters.append((cluster_han, cluster_viet, cluster_han_indices))

            combined_new_pairs = []
            
            for sub_han, sub_viet, sub_han_indices in sub_clusters:
                print(
                    f"[QwenRealign]   Processing sub-cluster with {len(sub_han)} Han, {len(sub_viet)} Viet..."
                )
                prompt = self._build_realign_prompt(sub_han, sub_viet)
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                with torch.no_grad():
                    output_tokens = self._model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                input_len = inputs.input_ids.shape[1]
                response_text = self._tokenizer.decode(
                    output_tokens[0][input_len:], skip_special_tokens=True
                )

                new_pairs = self._parse_realign_response(response_text, sub_han, sub_viet, sub_han_indices)
                if new_pairs is None:
                    raise ValueError(
                        f"QwenRealign failed to parse LLM output for sub-cluster.\n"
                        f"Han sentences: {sub_han}\n"
                        f"Viet sentences: {sub_viet}\n"
                        f"Raw Qwen response: {repr(response_text)}"
                    )
                
                combined_new_pairs.extend(new_pairs)

            # Replace the cluster slice with the combined re-aligned and/or fallback pairs
            aligned_pairs[cluster_start:cluster_end] = combined_new_pairs
            total_fixed += 1
            print(
                f"[QwenRealign] Cluster [{cluster_start}:{cluster_end}] replaced with "
                f"{len(combined_new_pairs)} pair(s)."
            )

        print(
            f"[QwenRealign] Phase 3 complete. "
            f"Fixed {total_fixed}/{len(clusters)} cluster(s)."
        )
        return aligned_pairs
