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
        self.model_name = self.config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
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

    def _load_model(self):
        """Lazy load Qwen model and tokenizer."""
        if self._model is not None:
            return

        print(f"[Qwen] Loading {self.model_name} (4-bit={self.load_in_4bit})...")
        t0 = time.time()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {"device_map": self.device_map, "torch_dtype": torch.float16}
        if self.load_in_4bit:
            # Requires bitsandbytes and accelerate packages
            try:
                import accelerate
                import bitsandbytes
            except ImportError:
                print(
                    "[Qwen] Warning: bitsandbytes or accelerate not installed. "
                    "Attempting to load model without 4-bit quantization."
                )
                self.load_in_4bit = False

        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        print(f"[Qwen] Model loaded. ({time.time() - t0:.1f}s)")

    def _build_prompt(self, han: str, viet: str) -> str:
        """Create the zero-shot evaluation prompt for Qwen."""
        system_content = "Bạn là chuyên gia Hán Nôm và dịch thuật cổ văn Việt Nam."
        user_content = f"""Hãy đánh giá chất lượng dóng hàng giữa câu chữ Hán và câu tiếng Việt dưới đây.
Chỉ trả lời bằng một số nguyên từ 0 đến 5:
  0 = Hoàn toàn không liên quan
  1 = Rất ít liên quan
  2 = Liên quan nhưng dịch sai nghĩa
  3 = Liên quan, dịch đúng nhưng không đầy đủ
  4 = Dịch đúng và gần đầy đủ
  5 = Dịch chính xác và đầy đủ ý nghĩa

Câu Hán: {han}
Câu Việt: {viet}

Điểm:"""

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
        # Find any digit in the response, prefer the first one
        match = re.search(r"\b([0-5])\b", text.strip())
        if match:
            return int(match.group(1))
        # Fallback search for any digit
        match_any = re.search(r"(\d)", text.strip())
        if match_any:
            val = int(match_any.group(1))
            return min(max(val, 0), 5)
        # Default fallback to 0 (reject if LLM output is garbled)
        return 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def verify(self, aligned_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify aligned pairs. Filter out candidates based on Qwen's score.

        Args:
            aligned_pairs: List of dicts representing aligned pairs. Each must contain:
                           "han_sentence", "viet_sentence", "similarity_score"

        Returns:
            List of dicts with additional keys:
               "qwen_score": int (0-5) or None (if skipped)
               "verified": bool (True/False)
        """
        if not aligned_pairs:
            return []

        # Find pairs that need verification (score in the uncertain range)
        uncertain_pairs_indices = []
        for idx, pair in enumerate(aligned_pairs):
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

        # Initialize all as verified (default) and qwen_score as None (skipped)
        for pair in aligned_pairs:
            pair["qwen_score"] = None
            pair["verified"] = True

        if total_uncertain == 0:
            print("[Qwen] No pairs need LLM verification. Skipping Qwen Phase 2.")
            return aligned_pairs

        # Load Qwen model (lazily)
        self._load_model()

        print(f"[Qwen] Running batch verification (batch_size={self.batch_size})...")
        t0 = time.time()
        processed = 0

        # Batch processing
        for batch_start in range(0, total_uncertain, self.batch_size):
            batch_indices = uncertain_pairs_indices[batch_start : batch_start + self.batch_size]
            prompts = []
            for idx in batch_indices:
                pair = aligned_pairs[idx]
                prompts.append(self._build_prompt(pair["han_sentence"], pair["viet_sentence"]))

            # Tokenize batch
            inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._model.device)

            # Generate
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,  # deterministic greedy decoding for rating
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode responses
            input_len = inputs.input_ids.shape[1]
            for i, idx in enumerate(batch_indices):
                response_tokens = outputs[i][input_len:]
                response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
                score = self._parse_score(response_text)

                # Set results
                aligned_pairs[idx]["qwen_score"] = score
                aligned_pairs[idx]["verified"] = (score >= self.keep_threshold)

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
