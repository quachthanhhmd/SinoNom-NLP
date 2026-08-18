"""Fail-fast Gemini credential check used by the Kaggle notebooks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple


def mask_key(key: str) -> str:
    key = str(key or "").strip()
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}…{key[-4:]}"


def interpret_response(status_code: int, body: str) -> Tuple[bool, str]:
    """Map an HTTP result to an actionable, secret-safe diagnostic."""
    if status_code == 200:
        return True, "OK"
    messages = {
        400: "request/model không hợp lệ",
        401: "API key không được xác thực",
        403: "API key sai, bị chặn, hoặc chưa được phép dùng Gemini API",
        404: "model Gemini cấu hình không tồn tại hoặc không khả dụng",
        429: "key đã hết quota hoặc đang bị rate-limit",
    }
    message = messages.get(status_code, f"Gemini trả HTTP {status_code}")
    try:
        payload = json.loads(body or "{}")
        detail = payload.get("error", {}).get("message", "")
    except (TypeError, ValueError, json.JSONDecodeError):
        detail = ""
    if detail:
        message = f"{message}: {str(detail)[:300]}"
    return False, message


def check_keys(keys: List[str], model: str, timeout: int = 30) -> Dict[str, object]:
    import requests

    results = []
    for index, key in enumerate(keys, start=1):
        masked = mask_key(key)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}"
        )
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": "Reply with exactly OK"}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8},
                },
                timeout=timeout,
            )
            usable, message = interpret_response(response.status_code, response.text)
            status_code = response.status_code
        except requests.RequestException as error:
            usable = False
            status_code = None
            message = f"lỗi kết nối tới Gemini: {error}"

        results.append({
            "index": index,
            "masked_key": masked,
            "usable": usable,
            "status_code": status_code,
            "message": message,
        })
    return {
        "model": model,
        "checked": len(results),
        "usable": sum(bool(item["usable"]) for item in results),
        "failed": sum(not bool(item["usable"]) for item in results),
        "results": results,
    }


def main() -> int:
    from config import ENSEMBLE_CONFIG

    parser = argparse.ArgumentParser(description="Validate Gemini keys before alignment")
    default_model = ENSEMBLE_CONFIG.get("qwen_verifier", {}).get(
        "realigner_model_name", "gemini-3.1-flash-lite"
    )
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    raw_keys = os.environ.get("GEMINI_API_KEY", "")
    keys = [item.strip() for item in raw_keys.split(",") if item.strip()]
    if not keys:
        print(
            "❌ GEMINI_API_KEY đang thiếu hoặc rỗng. Hãy tạo Kaggle Secret "
            "tên GEMINI_API_KEY rồi chạy lại cell kiểm tra.",
            file=sys.stderr,
        )
        return 2

    print(f"Đang kiểm tra {len(keys)} Gemini key với model {args.model}...")
    report = check_keys(keys, args.model, timeout=args.timeout)
    for item in report["results"]:
        symbol = "✅" if item["usable"] else "❌"
        print(
            f"{symbol} Key #{item['index']} ({item['masked_key']}): "
            f"{item['message']}"
        )

    if report["usable"] == 0:
        print(
            "❌ Không có Gemini key nào dùng được. Dừng notebook trước Phase 1/2/3 "
            "để tránh chạy nhiều giờ rồi mới lỗi.",
            file=sys.stderr,
        )
        return 3

    if report["failed"]:
        print(
            f"⚠️ Có {report['failed']} key lỗi nhưng còn {report['usable']} key dùng được; "
            "pipeline có thể tiếp tục và sẽ xoay qua các key hợp lệ."
        )
    else:
        print("✅ Tất cả Gemini API key đều hoạt động. Có thể chạy pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
