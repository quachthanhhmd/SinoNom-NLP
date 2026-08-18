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
from nlp.alignment_core import validate_alignment_records


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
        self.model_name = self.config.get("realigner_model_name") or self.config.get("model_name", "gemini-3.1-flash-lite")
        self.load_in_4bit = self.config.get("load_in_4bit", True)
        self.device_map = self.config.get("device_map", "auto")
        self.batch_size = self.config.get("batch_size", 4)

        # Allow reusing model loaded from QwenVerifier
        self._model = model
        self._tokenizer = tokenizer

    # Model loading
    # ------------------------------------------------------------------ #

    def _call_gemini(self, prompt: str, is_json: bool = False) -> str:
        import requests
        import os
        import json
        
        # Load API keys
        raw_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        if not raw_key:
            raise ValueError(
                "Gemini API key is missing. Please set GEMINI_API_KEY environment variable "
                "or specify 'api_key' in config.py under qwen_verifier."
            )
            
        if isinstance(raw_key, list):
            keys = raw_key
        else:
            keys = [k.strip() for k in raw_key.split(",") if k.strip()]
            
        if not keys:
            raise ValueError("No valid Gemini API keys found.")
            
        if not hasattr(self, "_current_key_idx") or self._current_key_idx >= len(keys):
            self._current_key_idx = 0
            
        model = self.model_name
        # Tăng số lượt thử lên gấp 5 lần số keys để có nhiều vòng xoay hơn
        max_retries = len(keys) * 5
        
        # Lưu các key bị lỗi 403 vĩnh viễn để tránh lặp lại vô ích
        bad_keys = set()
        
        for attempt in range(max_retries):
            # Nếu tất cả các keys đều hỏng (403), dừng ngay lập tức
            if len(bad_keys) >= len(keys):
                raise ValueError("Tất cả các Gemini API Keys của bạn đều bị lỗi 403 (Forbidden). Vui lòng kiểm tra lại Kaggle Secrets.")
                
            api_key = keys[self._current_key_idx]
            if api_key in bad_keys:
                self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                continue
                
            # Cứ mỗi lần xoay hết một vòng toàn bộ các key, cho hệ thống nghỉ 15 giây để hồi lại quota 429
            if attempt > 0 and attempt % len(keys) == 0:
                print(f"[Gemini] Đã thử qua một vòng tất cả các key. Tạm dừng 15 giây để hồi lại hạn mức (quota)...")
                time.sleep(15.0)
                
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.0
                }
            }
            if is_json:
                payload["generationConfig"]["responseMimeType"] = "application/json"
                
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return result['candidates'][0]['content']['parts'][0]['text']
                elif response.status_code == 429:
                    old_idx = self._current_key_idx
                    self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                    print(f"[Gemini] Key #{old_idx+1} bị hết hạn mức hoặc rate limit (429). Đang chuyển sang Key #{self._current_key_idx+1}...")
                    time.sleep(1.0)
                elif response.status_code == 403:
                    old_idx = self._current_key_idx
                    bad_keys.add(api_key)
                    self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                    print(f"[Gemini] ⚠️ Lỗi 403 (Forbidden) với Key #{old_idx+1}. Key này không hợp lệ hoặc sai định dạng. Đã loại bỏ key này khỏi vòng xoay.")
                else:
                    old_idx = self._current_key_idx
                    self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                    print(f"[Gemini] Key #{old_idx+1} trả về lỗi HTTP {response.status_code}. Đang xoay sang Key #{self._current_key_idx+1}...")
                    time.sleep(1.0)
            except Exception as e:
                old_idx = self._current_key_idx
                self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                print(f"[Gemini] Lỗi kết nối với Key #{old_idx+1}: {e}. Đang xoay sang Key #{self._current_key_idx+1}...")
                time.sleep(1.0)
                
        raise ValueError("Không thể gọi Gemini API sau khi đã thử tất cả các key và áp dụng thời gian chờ (rate limit).")

    def _load_model(self):
        if self.model_name.lower().startswith("gemini"):
            return
            
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

    @staticmethod
    def _build_example_json(num_han: int, num_viet: int) -> str:
        """Return a schema example that never cites a nonexistent H/V ID."""
        example = []
        paired = min(num_han, num_viet)
        for index in range(paired):
            example.append({"han": f"H{index + 1}", "viet": f"V{index + 1}"})
        for index in range(paired, num_han):
            example.append({"han": f"H{index + 1}", "viet": None})
        for index in range(paired, num_viet):
            example.append({"han": None, "viet": f"V{index + 1}"})
        return json.dumps(example, ensure_ascii=False, indent=2)

    def _build_realign_prompt(self, han_sentences: List[str], viet_sentences: List[str]) -> str:
        num_han = len(han_sentences)
        num_viet = len(viet_sentences)
        
        han_numbered = "\n".join(f"  H{i+1}: {s}" for i, s in enumerate(han_sentences))
        viet_numbered = "\n".join(f"  V{j+1}: {s}" for j, s in enumerate(viet_sentences))

        system_content = (
            "Bạn là chuyên gia Hán Nôm và dịch thuật cổ văn Việt Nam với kinh nghiệm sâu rộng "
            "về Đại Nam Nhất Thống Chí."
        )
        
        example_json = self._build_example_json(num_han, num_viet)

        user_content = f"""Dưới đây là các câu chữ Hán cổ và các câu dịch tiếng Việt tương ứng bị lệch pha (không được dóng hàng đúng chỗ).

Nhiệm vụ của bạn: tạo được NHIỀU BEAD EXACT NHẤT CÓ THỂ. Mỗi bead phải tương ứng đầy đủ nội dung; bead có addition hoặc omission là sai.

QUY TẮC BẮT BUỘC:
1. Chỉ được dùng các mã Hán từ H1 đến H{num_han} và các mã Việt từ V1 đến V{num_viet} (dựa theo danh sách CÁC CÂU HÁN và CÁC CÂU VIỆT bên dưới).
2. TUYỆT ĐỐI KHÔNG sử dụng các mã câu không tồn tại trong danh sách (Ví dụ: nếu danh sách chỉ có V1 thì không được dùng V2, V3...).
3. Mỗi cặp JSON phải có "han" (chứa mã tham chiếu câu Hán, ví dụ: "H1", hoặc nhiều câu gộp như "H1+H2") và "viet" (chứa mã tham chiếu câu Việt, ví dụ: "V1", hoặc nhiều câu gộp như "V1+V2").
4. BẮT BUỘC dùng mã số thứ tự H1, H2... cho Hán và V1, V2... cho Việt. KHÔNG ĐƯỢC tự ý viết lại toàn bộ nội dung văn bản gốc vào JSON.
5. Mỗi câu CHỈ được dùng một lần.
6. Nếu một câu Hán không tìm được câu Việt phù hợp, đặt "viet": null.
7. Nếu một câu Việt không tìm được câu Hán phù hợp, đặt "han": null.
8. Trả lời ĐÚNG định dạng JSON, không giải thích thêm.
9. Được phép ghép m-n để sửa lệch boundary. Chỉ ghép các mã liên tiếp và không ghép thêm nội dung không tương ứng.
10. Thà đặt null phần chưa chắc chắn còn hơn tạo một bead chỉ khớp một phần.

CÁC CÂU HÁN:
{han_numbered}

CÁC CÂU VIỆT:
{viet_numbered}

Ví dụ kết quả dóng hàng mong muốn cho cụm này (JSON array):
{example_json}
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
        global_han_indices: List[List[int]],
        global_viet_indices: List[List[int]],
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
        used_han_refs = []
        used_viet_refs = []
        for item in raw_pairs:
            if not isinstance(item, dict):
                continue

            # Resolve "H1", "H2"... references to actual text
            han_val = item.get("han")
            viet_val = item.get("viet")

            han_refs = self._resolve_refs(han_val, len(han_sentences), "H")
            viet_refs = self._resolve_refs(viet_val, len(viet_sentences), "V")
            if han_refs is None or viet_refs is None or (not han_refs and not viet_refs):
                return None
            if han_refs and han_refs != list(range(han_refs[0], han_refs[-1] + 1)):
                return None
            if viet_refs and viet_refs != list(range(viet_refs[0], viet_refs[-1] + 1)):
                return None
            used_han_refs.extend(han_refs)
            used_viet_refs.extend(viet_refs)
            han_text = " ".join(han_sentences[index] for index in han_refs)
            viet_text = " ".join(viet_sentences[index] for index in viet_refs)

            # Resolve references to actual index list
            resolved_han_indices = [
                source_index for ref in han_refs for source_index in global_han_indices[ref]
            ]
            resolved_viet_indices = [
                source_index for ref in viet_refs for source_index in global_viet_indices[ref]
            ]

            if han_text or viet_text:
                results.append({
                    "han_sentence": han_text or "",
                    "viet_sentence": viet_text or "",
                    # LLM proposals are routed to review.  Do not fabricate a
                    # semantic score or promote them to accepted automatically.
                    "similarity_score": 0.0,
                    "qwen_realigned": True,
                    "han_indices": resolved_han_indices,
                    "viet_indices": resolved_viet_indices,
                    "confidence": 0.0,
                    "status": "review" if (han_text and viet_text) else "unmatched",
                    "completeness_label": None if (han_text and viet_text) else "unmatched",
                })

        if used_han_refs != list(range(len(han_sentences))):
            return None
        if used_viet_refs != list(range(len(viet_sentences))):
            return None
        try:
            validate_alignment_records(results)
        except ValueError:
            return None
        return results if results else None

    def _resolve_refs(self, val: Any, sentence_count: int, prefix: str) -> Optional[List[int]]:
        """Accept only explicit H/V references; reject copied or invented text."""
        if val is None or str(val).strip().lower() == "null":
            return []
        value = str(val).strip()
        refs = [int(item) - 1 for item in re.findall(rf"{prefix}(\d+)", value, re.IGNORECASE)]
        residue = re.sub(rf"{prefix}\d+", "", value, flags=re.IGNORECASE)
        residue = re.sub(r"[+\s,;]", "", residue)
        if residue or not refs or any(index < 0 or index >= sentence_count for index in refs):
            return None
        if refs != sorted(set(refs)):
            return None
        return refs

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

        # Plain text is never trusted: the model must cite source references.
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def realign(
        self, aligned_pairs: List[Dict[str, Any]], cache_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: Detect NaN clusters and re-align them using Qwen/Gemini.

        Args:
            aligned_pairs: List of dicts from Phase 2 output.
            cache_path: Optional path to save intermediate/final Phase 3 JSON output.

        Returns:
            Updated list with NaN clusters replaced by Qwen's re-alignments.
        """
        import json
        import os

        original_han_indices = [
            index for pair in aligned_pairs for index in pair.get("han_indices", [])
        ]
        original_viet_indices = [
            index for pair in aligned_pairs for index in pair.get("viet_indices", [])
        ]

        # Resume from checkpoint if cache_path exists
        if cache_path and os.path.exists(cache_path):
            print(f"[Cache] Found Phase 3 checkpoint/cached alignment at: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data:
                    aligned_pairs = cached_data
                    print(f"[Cache] Successfully loaded Phase 3 checkpoint. Resuming...")
            except Exception as e:
                print(f"[Cache] Warning: Failed to load Phase 3 cache: {e}. Starting fresh...")

        clusters = self._detect_nan_clusters(aligned_pairs)
        if not clusters:
            print("[QwenRealign] No NaN clusters found. Phase 3 completed/skipped.")
            expected_han_count = max(original_han_indices) + 1 if original_han_indices else 0
            expected_viet_count = max(original_viet_indices) + 1 if original_viet_indices else 0
            validate_alignment_records(aligned_pairs, expected_han_count, expected_viet_count)
            return aligned_pairs

        print(f"[QwenRealign] Found {len(clusters)} NaN cluster(s) to re-align.")
        is_gemini = self.model_name.lower().startswith("gemini")
        if not is_gemini:
            self._load_model()

        # Clear VRAM before starting realignments (only for local model)
        if not is_gemini:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

        total_fixed = 0
        # Process clusters in reverse order so index replacement doesn't shift positions
        for cluster_start, cluster_end, cluster_han, cluster_viet in reversed(clusters):
            # Extract Han indices for the cluster (preserving 1-to-1 mapping even for merged sentences)
            cluster_han_indices = []
            cluster_viet_indices = []
            for item in aligned_pairs[cluster_start:cluster_end]:
                h = str(item.get("han_sentence", "") or "").strip()
                h_is_nan = not h or h.lower() == "nan"
                if not h_is_nan:
                    cluster_han_indices.append(item.get("han_indices", []))
                v = str(item.get("viet_sentence", "") or "").strip()
                if v and v.lower() != "nan":
                    cluster_viet_indices.append(item.get("viet_indices", []))

            # Gemini can inspect a wider local block without VRAM pressure. The
            # checked corpus has 99%+ of unresolved clusters within 60 source
            # rows; handling that window directly recovers substantially more
            # exact beads without reintroducing unsafe proportional slicing.
            limit = 60 if is_gemini else 12
            if len(cluster_han) > limit or len(cluster_viet) > limit:
                print(
                    f"[QwenRealign] Cluster [{cluster_start}:{cluster_end}] exceeds the safe "
                    f"window ({len(cluster_han)} Han, {len(cluster_viet)} Viet). "
                    "Leaving it unmatched for review instead of guessing a proportional split."
                )
                continue
            sub_clusters = [(cluster_han, cluster_viet, cluster_han_indices, cluster_viet_indices)]

            combined_new_pairs = []
            
            for sub_han, sub_viet, sub_han_indices, sub_viet_indices in sub_clusters:
                print(
                    f"[QwenRealign]   Processing sub-cluster with {len(sub_han)} Han, {len(sub_viet)} Viet..."
                )
                if is_gemini:
                    num_han = len(sub_han)
                    num_viet = len(sub_viet)
                    han_numbered = "\n".join(f"  H{i+1}: {s}" for i, s in enumerate(sub_han))
                    viet_numbered = "\n".join(f"  V{j+1}: {s}" for j, s in enumerate(sub_viet))

                    example_json = self._build_example_json(num_han, num_viet)

                    system_content = "Bạn là chuyên gia Hán Nôm và dịch thuật cổ văn Việt Nam với kinh nghiệm sâu rộng về Đại Nam Nhất Thống Chí."
                    user_content = f"""Dưới đây là các câu chữ Hán cổ và các câu dịch tiếng Việt tương ứng bị lệch pha (không được dóng hàng đúng chỗ).

Nhiệm vụ của bạn: tạo được NHIỀU BEAD EXACT NHẤT CÓ THỂ. Mỗi bead phải tương ứng đầy đủ nội dung; bead có addition hoặc omission là sai.

QUY TẮC BẮT BUỘC:
1. Chỉ được dùng các mã Hán từ H1 đến H{num_han} và các mã Việt từ V1 đến V{num_viet} (dựa theo danh sách CÁC CÂU HÁN và CÁC CÂU VIỆT bên dưới).
2. TUYỆT ĐỐI KHÔNG sử dụng các mã câu không tồn tại trong danh sách (Ví dụ: nếu danh sách chỉ có V1 thì không được dùng V2, V3...).
3. Mỗi cặp JSON phải có "han" (chứa mã tham chiếu câu Hán, ví dụ: "H1", hoặc nhiều câu gộp như "H1+H2") và "viet" (chứa mã tham chiếu câu Việt, ví dụ: "V1", hoặc nhiều câu gộp như "V1+V2").
4. BẮT BUỘC dùng mã số thứ tự H1, H2... cho Hán và V1, V2... cho Việt. KHÔNG ĐƯỢC tự ý viết lại toàn bộ nội dung văn bản gốc vào JSON.
5. Mỗi câu CHỈ được dùng một lần.
6. Nếu một câu Hán không tìm được câu Việt phù hợp, đặt "viet": null.
7. Nếu một câu Việt không tìm được câu Hán phù hợp, đặt "han": null.
8. Trả lời ĐÚNG định dạng JSON, không giải thích thêm.
9. Được phép ghép m-n để sửa lệch boundary. Chỉ ghép các mã liên tiếp và không ghép thêm nội dung không tương ứng.
10. Thà đặt null phần chưa chắc chắn còn hơn tạo một bead chỉ khớp một phần.

CÁC CÂU HÁN:
{han_numbered}

CÁC CÂU VIỆT:
{viet_numbered}

Ví dụ kết quả dóng hàng mong muốn cho cụm này (JSON array):
{example_json}
"""
                    prompt = f"{system_content}\n\n{user_content}"
                    response_text = self._call_gemini(prompt, is_json=True)
                else:
                    prompt = self._build_realign_prompt(sub_han, sub_viet)
                    import torch
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

                new_pairs = self._parse_realign_response(
                    response_text,
                    sub_han,
                    sub_viet,
                    sub_han_indices,
                    sub_viet_indices,
                )
                if new_pairs is None:
                    print(
                        "[QwenRealign] Rejected invalid LLM proposal; cluster remains "
                        "unmatched for review."
                    )
                    combined_new_pairs = []
                    break
                
                combined_new_pairs.extend(new_pairs)
                if is_gemini:
                    time.sleep(2.0)

            if not combined_new_pairs:
                continue

            # Replace only a complete, validated proposal.
            aligned_pairs[cluster_start:cluster_end] = combined_new_pairs
            total_fixed += 1
            print(
                f"[QwenRealign] Cluster [{cluster_start}:{cluster_end}] replaced with "
                f"{len(combined_new_pairs)} pair(s)."
            )
            
            # Save progress to checkpoint cache
            if cache_path:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)
                    print(f"[Cache] Saved Phase 3 checkpoint to: {cache_path}")
                except Exception as e:
                    print(f"[Cache] Warning: Failed to save Phase 3 checkpoint: {e}")

        print(
            f"[QwenRealign] Phase 3 complete. "
            f"Fixed {total_fixed}/{len(clusters)} cluster(s)."
        )
        expected_han_count = max(original_han_indices) + 1 if original_han_indices else 0
        expected_viet_count = max(original_viet_indices) + 1 if original_viet_indices else 0
        validate_alignment_records(aligned_pairs, expected_han_count, expected_viet_count)
        return aligned_pairs
