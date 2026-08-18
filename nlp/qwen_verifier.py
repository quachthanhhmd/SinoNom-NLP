"""Strict completeness verification for Hán--Việt alignment beads.

Phase 1 similarity is useful for proposing boundaries, but it is not evidence
that a bead is complete. This verifier checks *every* two-sided bead and labels
it using the evaluator's exact/addition/omission/mismatch rubric.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from config import ENSEMBLE_CONFIG
from nlp.bead_quality import COMPLETENESS_LABELS, bead_key, has_text, normalize_label


class QwenVerifier:
    """Offline Qwen verifier, with an optional Gemini-compatible backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ENSEMBLE_CONFIG.get("qwen_verifier", {})
        self.model_name = self.config.get("verifier_model_name") or self.config.get(
            "model_name", "Qwen/Qwen2.5-7B-Instruct"
        )
        self.load_in_4bit = self.config.get("load_in_4bit", True)
        self.device_map = self.config.get("device_map", "auto")
        self.batch_size = int(self.config.get("batch_size", 8))
        self.max_new_tokens = int(self.config.get("verification_max_new_tokens", 96))
        self.checkpoint_interval = max(
            self.batch_size, int(self.config.get("verification_checkpoint_interval", 256))
        )
        self._model = None
        self._tokenizer = None

    @staticmethod
    def _prompt_content(han: str, viet: str) -> str:
        return f"""Đánh giá một bead Hán–Việt theo độ ĐẦY ĐỦ NỘI DUNG, không chỉ theo độ tương đồng.

Quy ước bắt buộc (xem tiếng Việt là bản dịch của Hán):
- exact: hai phía tương ứng đầy đủ mọi ý, tên riêng, con số, quan hệ và phạm vi. Khác cách diễn đạt được phép; không được dư hay thiếu thông tin.
- addition: phía Việt có nội dung thêm không xuất hiện trong Hán.
- omission: phía Việt thiếu nội dung có trong Hán.
- mismatch: nội dung chính khác nhau, lệch câu/block, hoặc vừa thêm vừa thiếu nên không thể coi là một bản dịch đầy đủ.

Không được chấm exact nếu một câu chỉ chứa một phần nội dung của câu kia. Một chi tiết địa danh, chức tước, số liệu, hướng, khoảng cách hoặc mệnh đề bị dư/thiếu cũng là lỗi.

HÁN: {han}
VIỆT: {viet}

Chỉ trả về một JSON object đúng schema sau, không markdown:
{{"label":"exact|addition|omission|mismatch","extra_side":"none|han|viet|both","missing_side":"none|han|viet|both","confidence":0.0,"reason":"lý do ngắn"}}"""

    def _build_prompt(self, han: str, viet: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia Hán Nôm, chuyên kiểm định parallel corpus theo "
                    "từng bead. Ưu tiên tính đầy đủ nội dung và phải phân biệt addition, "
                    "omission, mismatch với exact."
                ),
            },
            {"role": "user", "content": self._prompt_content(han, viet)},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _parse_result(text: str) -> Dict[str, Any]:
        """Parse one model response; malformed responses fail closed."""
        clean = str(text or "").strip()
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
        payload: Dict[str, Any] = {}

        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
            try:
                decoded = json.loads(candidate)
                if isinstance(decoded, dict):
                    payload = decoded
            except json.JSONDecodeError:
                payload = {}

        if payload:
            raw_label = payload.get("label") or payload.get("result")
        else:
            label_match = re.search(
                r"\b(exact|addition|omission|mismatch|match|correct|extra|missing|wrong|unrelated)\b",
                clean,
                flags=re.IGNORECASE,
            )
            raw_label = label_match.group(1) if label_match else "mismatch"

        label = normalize_label(raw_label)
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 1.0 and confidence <= 100.0:
            confidence /= 100.0
        confidence = min(1.0, max(0.0, confidence))

        valid_sides = {"none", "han", "viet", "both"}
        extra_side = str(payload.get("extra_side", "none")).strip().lower()
        missing_side = str(payload.get("missing_side", "none")).strip().lower()
        if extra_side not in valid_sides:
            extra_side = "none"
        if missing_side not in valid_sides:
            missing_side = "none"

        return {
            "label": label,
            "extra_side": extra_side,
            "missing_side": missing_side,
            "confidence": confidence,
            "reason": str(payload.get("reason", "model output could not be fully parsed"))[:500],
            "raw_response": clean[:2000],
        }

    def _call_gemini(self, prompt: str) -> str:
        import requests

        raw_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        keys = raw_key if isinstance(raw_key, list) else str(raw_key).split(",")
        keys = [str(key).strip() for key in keys if str(key).strip()]
        if not keys:
            raise ValueError("GEMINI_API_KEY is required for a Gemini verifier model.")

        if not hasattr(self, "_current_key_idx"):
            self._current_key_idx = 0
        bad_keys = set()
        for attempt in range(max(5, len(keys) * 5)):
            if len(bad_keys) == len(keys):
                break
            key = keys[self._current_key_idx % len(keys)]
            self._current_key_idx = (self._current_key_idx + 1) % len(keys)
            if key in bad_keys:
                continue
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model_name}:generateContent?key={key}"
            )
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            if response.status_code == 403:
                bad_keys.add(key)
            elif response.status_code == 429:
                time.sleep(2.0)
            else:
                time.sleep(1.0)
            if attempt and attempt % len(keys) == 0:
                time.sleep(5.0)
        raise ValueError("Gemini verification failed after rotating all configured API keys.")

    def _load_model(self):
        if self.model_name.lower().startswith("gemini") or self._model is not None:
            return

        print(f"[Completeness] Loading {self.model_name} (4-bit={self.load_in_4bit})...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"device_map": self.device_map}
        if self.load_in_4bit:
            try:
                import accelerate  # noqa: F401
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                self.load_in_4bit = False
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

    @staticmethod
    def _apply_result(pair: Dict[str, Any], result: Dict[str, Any]) -> None:
        label = normalize_label(result.get("label"))
        pair["completeness_label"] = label
        pair["extra_side"] = result.get("extra_side", "none")
        pair["missing_side"] = result.get("missing_side", "none")
        pair["verification_confidence"] = result.get("confidence", 0.0)
        pair["verification_reason"] = result.get("reason", "")
        pair["verification_raw_response"] = result.get("raw_response", "")
        pair["verified"] = label == "exact"
        pair["status"] = "accepted" if label == "exact" else label
        # Compatibility only. Numeric scores are no longer an acceptance gate.
        pair["qwen_score"] = 5 if label == "exact" else 0

    @staticmethod
    def _save_checkpoint(path: Optional[str], records: List[Dict[str, Any]]) -> None:
        if not path:
            return
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def verify(
        self,
        aligned_pairs: List[Dict[str, Any]],
        cache_path: Optional[str] = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Classify every two-sided bead that has not already been classified."""
        if not aligned_pairs:
            return []

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                cache_map = {
                    bead_key(item): item
                    for item in cached
                    if item.get("completeness_label") in COMPLETENESS_LABELS
                }
                for pair in aligned_pairs:
                    cached_pair = cache_map.get(bead_key(pair))
                    if cached_pair:
                        for field in (
                            "completeness_label", "extra_side", "missing_side",
                            "verification_confidence", "verification_reason",
                            "verification_raw_response", "verified", "status", "qwen_score",
                        ):
                            if field in cached_pair:
                                pair[field] = cached_pair[field]
                print(f"[Completeness] Resumed {len(cache_map)} classified beads from checkpoint.")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"[Completeness] Ignoring unreadable checkpoint: {error}")

        pending: List[int] = []
        for index, pair in enumerate(aligned_pairs):
            if not has_text(pair.get("han_sentence")) or not has_text(pair.get("viet_sentence")):
                pair["completeness_label"] = "unmatched"
                pair["verified"] = False
                pair["status"] = "unmatched"
                continue
            if force or pair.get("completeness_label") not in COMPLETENESS_LABELS:
                pending.append(index)

        print(
            f"[Completeness] Verifying {len(pending)}/{len(aligned_pairs)} two-sided beads; "
            "no similarity-based auto-accept is allowed."
        )
        if not pending:
            return aligned_pairs

        is_gemini = self.model_name.lower().startswith("gemini")
        if is_gemini:
            for done, index in enumerate(pending, 1):
                pair = aligned_pairs[index]
                response = self._call_gemini(
                    self._prompt_content(pair["han_sentence"], pair["viet_sentence"])
                )
                self._apply_result(pair, self._parse_result(response))
                if done % 50 == 0 or done == len(pending):
                    print(f"[Completeness] Processed {done}/{len(pending)} beads...")
                if done % self.checkpoint_interval == 0 or done == len(pending):
                    self._save_checkpoint(cache_path, aligned_pairs)
            return aligned_pairs

        self._load_model()
        import torch

        processed = 0
        for start in range(0, len(pending), self.batch_size):
            batch_indices = pending[start : start + self.batch_size]
            prompts = [
                self._build_prompt(
                    aligned_pairs[index]["han_sentence"],
                    aligned_pairs[index]["viet_sentence"],
                )
                for index in batch_indices
            ]
            try:
                inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(
                    self._model.device
                )
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                input_length = inputs.input_ids.shape[1]
                for row, index in enumerate(batch_indices):
                    response = self._tokenizer.decode(
                        outputs[row][input_length:], skip_special_tokens=True
                    )
                    self._apply_result(aligned_pairs[index], self._parse_result(response))
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or len(batch_indices) == 1:
                    raise
                print("[Completeness] Batch OOM; retrying this batch one bead at a time.")
                torch.cuda.empty_cache()
                for index in batch_indices:
                    prompt = self._build_prompt(
                        aligned_pairs[index]["han_sentence"],
                        aligned_pairs[index]["viet_sentence"],
                    )
                    inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
                    with torch.no_grad():
                        output = self._model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self._tokenizer.eos_token_id,
                        )
                    response = self._tokenizer.decode(
                        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    self._apply_result(aligned_pairs[index], self._parse_result(response))

            processed += len(batch_indices)
            print(f"[Completeness] Processed {processed}/{len(pending)} beads...")
            if (
                processed % self.checkpoint_interval == 0
                or processed == len(pending)
            ):
                self._save_checkpoint(cache_path, aligned_pairs)

        exact = sum(
            pair.get("completeness_label") == "exact" for pair in aligned_pairs
        )
        print(f"[Completeness] Verification complete: {exact} exact beads.")
        return aligned_pairs

    def free_gpu_memory(self):
        import gc

        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
