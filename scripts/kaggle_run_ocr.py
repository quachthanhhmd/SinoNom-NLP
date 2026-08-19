"""
Kaggle GPU runner cho Sino-Nom OCR.

Dán từng CELL bên dưới vào Kaggle Notebook, hoặc chạy thẳng file này bằng
`!python scripts/kaggle_run_ocr.py --volume q1`.

YÊU CẦU TRÊN KAGGLE
  Settings -> Accelerator : GPU T4 x2  (hoặc P100)
  Settings -> Internet    : ON   (bắt buộc: pip install + tải model PaddleOCR)

CHÚ Ý paddlepaddle-gpu: bản trên PyPI KHÔNG có CUDA. Phải cài từ index của
Paddle, đúng phiên bản CUDA mà Kaggle đang dùng. Cell 1 tự dò.
"""
import argparse
import os
import subprocess
import sys
import time

REPO_DEFAULT = "/kaggle/working/sinonom-ocr"


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def detect_cuda_tag() -> str:
    """Trả về tag index của Paddle khớp CUDA đang có trên máy."""
    try:
        out = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True).stdout
        import re
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            if major >= 13:
                return "cu130"
            if major == 12:
                return "cu129" if minor >= 9 else ("cu126" if minor >= 6 else "cu118")
            return "cu118"
    except Exception:
        pass
    return "cu126"


def install_deps():
    tag = detect_cuda_tag()
    print(f"[deps] CUDA tag = {tag}")
    sh("pip uninstall -y paddlepaddle paddlepaddle-gpu >/dev/null 2>&1", check=False)
    sh(f"pip install -q paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/{tag}/")
    sh("pip install -q 'paddleocr>=3.7.0' scipy opencc-python-reimplemented opencv-python-headless")


def verify_gpu():
    import paddle
    print("paddle", paddle.__version__)
    ok = paddle.device.is_compiled_with_cuda()
    n = paddle.device.cuda.device_count() if ok else 0
    print(f"compiled_with_cuda={ok}  gpu_count={n}")
    if not ok or n == 0:
        print("!! Paddle KHÔNG thấy GPU -> sẽ chạy CPU (rất chậm).")
        print("   Kiểm tra Accelerator đã bật GPU chưa, và cell cài đặt có lỗi không.")
    return n


def run(repo: str, volume: str, input_dir: str, out: str, n_gpu: int, limit=None):
    os.chdir(repo)
    env = f"PYTHONPATH={repo}"
    extra = f" --limit {limit}" if limit else ""
    t0 = time.time()

    if n_gpu >= 2:
        # Mỗi GPU một tiến trình, chia trang xen kẽ. Chạy song song rồi gộp.
        procs = []
        for i in range(n_gpu):
            cmd = (f"{env} python scripts/run_han_ocr.py --input {input_dir} "
                   f"--out {out} --device gpu:{i} --shard {i}/{n_gpu}{extra}")
            print(f"$ {cmd} &", flush=True)
            procs.append(subprocess.Popen(cmd, shell=True))
        codes = [p.wait() for p in procs]
        if any(codes):
            print(f"!! có shard lỗi: {codes}")
        sh(f"{env} python scripts/merge_shards.py --out {out}")
    else:
        dev = "gpu:0" if n_gpu == 1 else "cpu"
        sh(f"{env} python scripts/run_han_ocr.py --input {input_dir} --out {out} --device {dev}{extra}")

    print(f"\nTổng thời gian: {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--volume", default="q1")
    ap.add_argument("--input", default=None, help="Mặc định <repo>/dataset/china/<volume>")
    ap.add_argument("--out", default=None, help="Mặc định /kaggle/working/<volume>.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    if not args.skip_install:
        install_deps()
    n_gpu = verify_gpu()

    input_dir = args.input or os.path.join(args.repo, "dataset", "china", args.volume)
    out = args.out or f"/kaggle/working/{args.volume}.json"
    run(args.repo, args.volume, input_dir, out, n_gpu, args.limit)


if __name__ == "__main__":
    main()
