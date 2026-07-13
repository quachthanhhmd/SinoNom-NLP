import os
import requests
import json
import time

def test_keys():
    raw_keys = os.environ.get("GEMINI_API_KEY", "")
    if not raw_keys:
        print("❌ [LỖI] Chưa nạp biến môi trường GEMINI_API_KEY. Vui lòng cấu hình trong Kaggle Secrets.")
        return

    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    print(f"🔎 Tìm thấy {len(keys)} API Key(s) trong cấu hình. Bắt đầu kiểm tra nhanh...")
    print("=" * 80)

    # Đọc model name từ cấu hình của hệ thống hoặc dùng mặc định
    model = "gemini-3.1-flash-lite"
    try:
        from config import ENSEMBLE_CONFIG
        if "qwen_verifier" in ENSEMBLE_CONFIG:
            model = ENSEMBLE_CONFIG["qwen_verifier"].get("realigner_model_name", "gemini-3.1-flash-lite")
    except Exception:
        pass

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": "Hi"}]}],
        "generationConfig": {"temperature": 0.0}
    }

    ok_count = 0
    bad_count = 0

    for idx, api_key in enumerate(keys, 1):
        # Che bớt key để bảo mật khi log
        masked_key = api_key[:10] + "..." + api_key[-5:] if len(api_key) > 15 else api_key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        try:
            t0 = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            elapsed = time.time() - t0
            
            if response.status_code == 200:
                print(f"✅ Key #{idx} ({masked_key}): HOẠT ĐỘNG TỐT (Phản hồi: {elapsed:.2f}s)")
                ok_count += 1
            elif response.status_code == 429:
                print(f"⚠️  Key #{idx} ({masked_key}): HỢP LỆ nhưng đang tạm bận/hết hạn mức phút (HTTP 429)")
                ok_count += 1 # Vẫn được tính là key hợp lệ
            elif response.status_code == 403:
                print(f"❌ Key #{idx} ({masked_key}): LỖI 403 (FORBIDDEN) - Key không hợp lệ hoặc chưa đồng ý điều khoản!")
                bad_count += 1
            else:
                print(f"❌ Key #{idx} ({masked_key}): LỖI HTTP {response.status_code} - {response.text[:100]}")
                bad_count += 1
        except Exception as e:
            print(f"❌ Key #{idx} ({masked_key}): LỖI KẾT NỐI - {e}")
            bad_count += 1

    print("=" * 80)
    print(f"📊 KẾT QUẢ KIỂM TRA: {ok_count} Key hoạt động tốt | {bad_count} Key bị lỗi.")
    if bad_count > 0:
        print("💡 Khuyên dùng: Hãy kiểm tra và tạo lại các Key bị báo lỗi 403 trên Google AI Studio.")

if __name__ == "__main__":
    test_keys()
