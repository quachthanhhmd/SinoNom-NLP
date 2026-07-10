"""
Qwen Verifier — Phase 2 post-processing using Qwen2.5-7B-Instruct.

Runs offline on Kaggle (using 4-bit quantization via bitsandbytes to fit in T4 GPU).
Filters out incorrect alignments by scoring them 0-5.
Only processes aligned pairs within the "uncertain zone" (e.g. ensemble score in [0.38, 0.50]).
"""

import re
import time
from typing import Any, Dict, List, Optional

from config import ENSEMBLE_CONFIG


class QwenVerifier:
    """
    Offline LLM verification filter using Qwen2.5-7B-Instruct.

    Only runs verification on candidate sentence pairs that fall into the
    similarity score uncertainty band. Other pairs are auto-approved.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Fallback to default config if none provided
        self.config = config or ENSEMBLE_CONFIG.get("qwen_verifier", {})
        self.model_name = self.config.get("verifier_model_name") or self.config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.load_in_4bit = self.config.get("load_in_4bit", True)
        self.device_map = self.config.get("device_map", "auto")
        self.uncertain_low = self.config.get("uncertain_low", 0.38)
        self.uncertain_high = self.config.get("uncertain_high", 0.50)
        self.keep_threshold = self.config.get("keep_threshold", 3)
        self.batch_size = self.config.get("batch_size", 8)

        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _call_gemini(self, prompt: str, is_json: bool = False) -> str:
        import requests
        import os
        import json
        import time
        
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
        max_retries = len(keys) * 2
        
        for attempt in range(max_retries):
            api_key = keys[self._current_key_idx]
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
                    if len(keys) > 1:
                        old_idx = self._current_key_idx
                        self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                        print(f"[Gemini] Key #{old_idx+1} rate-limited or exhausted (429). Rotating to Key #{self._current_key_idx+1}...")
                        time.sleep(1.0)
                    else:
                        sleep_time = (2 ** attempt) + 2
                        print(f"[Gemini] Rate limit (429) hit. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                else:
                    if len(keys) > 1:
                        old_idx = self._current_key_idx
                        self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                        print(f"[Gemini] Key #{old_idx+1} returned error {response.status_code}. Rotating to Key #{self._current_key_idx+1}...")
                        time.sleep(1.0)
                    else:
                        raise ValueError(f"Gemini API returned error {response.status_code}: {response.text}")
            except Exception as e:
                if len(keys) > 1:
                    old_idx = self._current_key_idx
                    self._current_key_idx = (self._current_key_idx + 1) % len(keys)
                    print(f"[Gemini] Exception with Key #{old_idx+1}: {e}. Rotating to Key #{self._current_key_idx+1}...")
                    time.sleep(1.0)
                else:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2)
        raise ValueError("Failed to call Gemini API after rotating all keys.")

    def _load_model(self):
        """Lazy load Qwen model and tokenizer."""
        if self.model_name.lower().startswith("gemini"):
            return
            
        if self._model is not None:
            return

        print(f"[Qwen] Loading {self.model_name} (4-bit={self.load_in_4bit})...")
        t0 = time.time()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = {"device_map": self.device_map}
        if self.load_in_4bit:
            # Requires bitsandbytes and accelerate packages
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
                print(
                    "[Qwen] Warning: bitsandbytes or accelerate not installed. "
                    "Attempting to load model without 4-bit quantization."
                )
                self.load_in_4bit = False
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        print(f"[Qwen] Model loaded. ({time.time() - t0:.1f}s)")

    def _build_prompt(self, han: str, viet: str) -> str:
        """Create the zero-shot evaluation prompt for Qwen."""
        system_content = "Bạn là chuyên gia Hán Nôm và dịch thuật cổ văn Việt Nam. Nhiệm vụ của bạn là đánh giá chất lượng dóng hàng để xây dựng tập dữ liệu song song sạch (Gold parallel corpus) cho dịch máy."
        user_content = f"""Hãy đánh giá xem câu tiếng Việt và câu chữ Hán dưới đây có phải là bản dịch sạch, khớp thông tin 1-1 trực tiếp hay không.
Chỉ trả lời bằng duy nhất một chữ số từ 0 đến 5, không giải thích gì thêm:
  5: Dịch chính xác, đầy đủ nghĩa, khớp thông tin trực tiếp 1-1, KHÔNG có chú thích dịch giả hay từ ngữ giải nghĩa thêm.
  4: Dịch đúng thông tin cốt lõi, khớp trực tiếp, có thể thừa/thiếu một vài trợ từ không quan trọng.
  3: Dịch đúng nhưng chứa thông tin thừa do dịch giả chú thích thêm trong ngoặc đơn (ví dụ: chú thích năm dương lịch, chú thích chữ Hán phụ) mà bản Hán gốc không có.
  2: Dịch thiếu rất nhiều thông tin cốt lõi hoặc chứa quá nhiều văn bản diễn giải dài dòng của dịch giả.
  1: Rất ít liên quan về mặt nội dung.
  0: Hoàn toàn không liên quan hoặc là hai câu khác nhau.

LƯU Ý QUAN TRỌNG: Để phục vụ huấn luyện dịch máy (Machine Translation), chúng ta cần tránh dữ liệu rác (hallucination). Vì vậy, các câu tiếng Việt có chứa chú thích của dịch giả trong ngoặc đơn hoặc diễn giải thêm mà bản Hán không có phải bị chấm điểm thấp (chấm 3 hoặc 2) để hệ thống tự động loại bỏ.

Câu Hán: {han}
Câu Việt: {viet}

Điểm số:"""

        # Format prompt using Qwen Chat Template
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _parse_score(self, text: str) -> int:
        """Extract a single integer score (0-5) from Qwen's response."""
        text_clean = text.strip()
        # Look for a digit at the very beginning of the response first
        match_start = re.match(r"^([0-5])", text_clean)
        if match_start:
            return int(match_start.group(1))
            
        # Fallback search for any digit in the response, prefer the first one
        match = re.search(r"\b([0-5])\b", text_clean)
        if match:
            return int(match.group(1))
        # Fallback search for any digit
        match_any = re.search(r"(\d)", text_clean)
        if match_any:
            val = int(match_any.group(1))
            return min(max(val, 0), 5)
        # Default fallback to 0 (reject if LLM output is garbled)
        return 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def verify(
        self, aligned_pairs: List[Dict[str, Any]], cache_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Verify aligned pairs. Filter out candidates based on Qwen's score.

        Args:
            aligned_pairs: List of dicts representing aligned pairs. Each must contain:
                           "han_sentence", "viet_sentence", "similarity_score"
            cache_path:    Optional path to save intermediate/final Phase 2 JSON output.

        Returns:
            List of dicts with additional keys:
               "qwen_score": int (0-5) or None (if skipped)
               "verified": bool (True/False)
        """
        if not aligned_pairs:
            return []

        # Check if cache_path exists and load it to resume progress
        import os
        import json
        if cache_path and os.path.exists(cache_path):
            print(f"[Cache] Found Phase 2 checkpoint/cached verification at: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data:
                    # Map cached verification status back to aligned_pairs
                    cache_map = {}
                    for item in cached_data:
                        k = (item.get("han_sentence", ""), item.get("viet_sentence", ""))
                        if item.get("qwen_score") is not None:
                            cache_map[k] = (item["qwen_score"], item["verified"])
                    
                    mapped_count = 0
                    for pair in aligned_pairs:
                        k = (pair.get("han_sentence", ""), pair.get("viet_sentence", ""))
                        if k in cache_map:
                            pair["qwen_score"], pair["verified"] = cache_map[k]
                            mapped_count += 1
                    print(f"[Cache] Successfully loaded Phase 2 checkpoint. Resumed {mapped_count} pairs.")
            except Exception as e:
                print(f"[Cache] Warning: Failed to load Phase 2 checkpoint: {e}")

        # Find pairs that need verification (score in the uncertain range and not yet verified)
        uncertain_pairs_indices = []
        for idx, pair in enumerate(aligned_pairs):
            if pair.get("qwen_score") is not None:
                continue
            score = pair.get("similarity_score", 0.0)
            # Only verify if both sentences are non-empty
            if not pair.get("han_sentence", "").strip() or not pair.get("viet_sentence", "").strip():
                continue
            if self.uncertain_low <= score <= self.uncertain_high:
                uncertain_pairs_indices.append(idx)

        total_uncertain = len(uncertain_pairs_indices)
        print(
            f"[Qwen] Found {total_uncertain}/{len(aligned_pairs)} pairs "
            f"in uncertainty band [{self.uncertain_low}, {self.uncertain_high}]."
        )

        # Initialize all as verified (default) and qwen_score as None (skipped) only if not set
        for pair in aligned_pairs:
            if "qwen_score" not in pair:
                pair["qwen_score"] = None
            if "verified" not in pair:
                pair["verified"] = True

        if total_uncertain == 0:
            print("[Qwen] No pairs need LLM verification. Skipping Qwen Phase 2.")
            return aligned_pairs

        # Load Qwen model (lazily) if not Gemini
        is_gemini = self.model_name.lower().startswith("gemini")
        if not is_gemini:
            self._load_model()

        t0 = time.time()
        processed = 0

        if is_gemini:
            print(f"[Gemini] Running verification sequentially (with 2.0s sleep to avoid rate limits)...")
            import time
            for idx in uncertain_pairs_indices:
                pair = aligned_pairs[idx]
                system_content = "Bạn là chuyên gia Hán Nôm và dịch thuật cổ văn Việt Nam. Nhiệm vụ của bạn là đánh giá chất lượng dóng hàng để xây dựng tập dữ liệu song song sạch (Gold parallel corpus) cho dịch máy."
                user_content = f"""Hãy đánh giá xem câu tiếng Việt và câu chữ Hán dưới đây có phải là bản dịch sạch, khớp thông tin 1-1 trực tiếp hay không.
Chỉ trả lời bằng duy nhất một chữ số từ 0 đến 5, không giải thích gì thêm:
  5: Dịch chính xác, đầy đủ nghĩa, khớp thông tin trực tiếp 1-1, KHÔNG có chú thích dịch giả hay từ ngữ giải nghĩa thêm.
  4: Dịch đúng thông tin cốt lõi, khớp trực tiếp, có thể thừa/thiếu một vài trợ từ không quan trọng.
  3: Dịch đúng nhưng chứa thông tin thừa do dịch giả chú thích thêm trong ngoặc đơn (ví dụ: chú thích năm dương lịch, chú thích chữ Hán phụ) mà bản Hán gốc không có.
  2: Dịch thiếu rất nhiều thông tin cốt lõi hoặc chứa quá nhiều văn bản diễn giải dài dòng của dịch giả.
  1: Rất ít liên quan về mặt nội dung.
  0: Hoàn toàn không liên quan hoặc là hai câu khác nhau.

LƯU Ý QUAN TRỌNG: Để phục vụ huấn luyện dịch máy (Machine Translation), chúng ta cần tránh dữ liệu rác (hallucination). Vì vậy, các câu tiếng Việt có chứa chú thích của dịch giả trong ngoặc đơn hoặc diễn giải thêm mà bản Hán không có phải bị chấm điểm thấp (chấm 3 hoặc 2) để hệ thống tự động loại bỏ.

Câu Hán: {pair["han_sentence"]}
Câu Việt: {pair["viet_sentence"]}

Điểm số:"""
                prompt = f"{system_content}\n\n{user_content}"
                response_text = self._call_gemini(prompt, is_json=False)
                score = self._parse_score(response_text)

                if idx in uncertain_pairs_indices[:5]:
                    print(f"[Gemini Debug] Cặp #{idx}:")
                    print(f"  Hán:  {repr(pair['han_sentence'])}")
                    print(f"  Việt: {repr(pair['viet_sentence'])}")
                    print(f"  Raw:  {repr(response_text)} -> Score parsed: {score}")

                aligned_pairs[idx]["qwen_score"] = score
                aligned_pairs[idx]["verified"] = (score >= self.keep_threshold)
                
                # Save progress to checkpoint
                if cache_path:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        pass

                processed += 1
                if processed < total_uncertain:
                    time.sleep(2.0)
            print(f"[Gemini] Processed {processed}/{total_uncertain} uncertain pairs...")
        else:
            print(f"[Qwen] Running batch verification (batch_size={self.batch_size})...")
            # Batch processing
            for batch_start in range(0, total_uncertain, self.batch_size):
                batch_indices = uncertain_pairs_indices[batch_start : batch_start + self.batch_size]
                prompts = []
                for idx in batch_indices:
                    pair = aligned_pairs[idx]
                    prompts.append(self._build_prompt(pair["han_sentence"], pair["viet_sentence"]))

                import torch
                try:
                    # Tokenize batch
                    inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._model.device)

                    # Generate
                    with torch.no_grad():
                        outputs = self._model.generate(
                            **inputs,
                            max_new_tokens=15,
                            do_sample=False,  # deterministic greedy decoding for rating
                            pad_token_id=self._tokenizer.eos_token_id,
                        )

                    # Decode responses
                    input_len = inputs.input_ids.shape[1]
                    for i, idx in enumerate(batch_indices):
                        response_tokens = outputs[i][input_len:]
                        response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
                        score = self._parse_score(response_text)

                        # In log debug cho 5 cặp đầu tiên của đợt xác thực
                        if idx in uncertain_pairs_indices[:5]:
                            print(f"[Qwen Debug] Cặp #{idx}:")
                            print(f"  Hán:  {repr(aligned_pairs[idx]['han_sentence'])}")
                            print(f"  Việt: {repr(aligned_pairs[idx]['viet_sentence'])}")
                            print(f"  Raw:  {repr(response_text)} -> Score parsed: {score}")

                        # Set results
                        aligned_pairs[idx]["qwen_score"] = score
                        aligned_pairs[idx]["verified"] = (score >= self.keep_threshold)

                    # Save progress to checkpoint
                    if cache_path:
                        try:
                            with open(cache_path, "w", encoding="utf-8") as f:
                                json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            pass

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[Qwen] Warning: CUDA Out of Memory with batch_size={len(prompts)}. Falling back to sequential execution (batch_size=1) for this batch...")
                        torch.cuda.empty_cache()
                        
                        # Process sequentially
                        for idx in batch_indices:
                            pair = aligned_pairs[idx]
                            p_prompt = self._build_prompt(pair["han_sentence"], pair["viet_sentence"])
                            p_inputs = self._tokenizer([p_prompt], return_tensors="pt").to(self._model.device)
                            
                            try:
                                with torch.no_grad():
                                    p_outputs = self._model.generate(
                                        **p_inputs,
                                        max_new_tokens=15,
                                        do_sample=False,
                                        pad_token_id=self._tokenizer.eos_token_id,
                                    )
                                p_input_len = p_inputs.input_ids.shape[1]
                                p_response_tokens = p_outputs[0][p_input_len:]
                                p_response_text = self._tokenizer.decode(p_response_tokens, skip_special_tokens=True)
                                score = self._parse_score(p_response_text)
                                
                                # In log debug cho chế độ tuần tự nếu thuộc 5 cặp đầu tiên
                                if idx in uncertain_pairs_indices[:5]:
                                    print(f"[Qwen Seq Debug] Cặp #{idx}:")
                                    print(f"  Hán:  {repr(pair['han_sentence'])}")
                                    print(f"  Việt: {repr(pair['viet_sentence'])}")
                                    print(f"  Raw:  {repr(p_response_text)} -> Score parsed: {score}")
                                
                                aligned_pairs[idx]["qwen_score"] = score
                                aligned_pairs[idx]["verified"] = (score >= self.keep_threshold)
                                
                                # Save progress to checkpoint
                                if cache_path:
                                    try:
                                        with open(cache_path, "w", encoding="utf-8") as f:
                                            json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)
                                    except Exception as e:
                                        pass
                            except RuntimeError as p_e:
                                if "CUDA out of memory" in str(p_e):
                                    print("[Qwen] Critical: CUDA Out of Memory even with batch_size=1! Skipping verification for this pair.")
                                    torch.cuda.empty_cache()
                                    aligned_pairs[idx]["qwen_score"] = 0
                                    aligned_pairs[idx]["verified"] = False
                                else:
                                    raise p_e
                    else:
                        raise e

                processed += len(batch_indices)
                print(f"[Qwen] Processed {processed}/{total_uncertain} uncertain pairs...")

        # Summarize results
        kept = sum(1 for idx in uncertain_pairs_indices if aligned_pairs[idx]["verified"])
        rejected = total_uncertain - kept
        print(
            f"[Qwen] Phase 2 Verification complete in {time.time() - t0:.1f}s. "
            f"Kept: {kept}, Rejected (score < {self.keep_threshold}): {rejected}."
        )

        return aligned_pairs
