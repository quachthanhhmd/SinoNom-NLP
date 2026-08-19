# Chạy Sino-Nom OCR trên Kaggle (GPU)

## Cài đặt notebook

| Setting | Giá trị |
|---|---|
| Accelerator | **GPU T4 x2** (script tự dùng cả 2) |
| Internet | **ON** — bắt buộc, để pip install và tải model PaddleOCR |
| Persistence | Variables and Files (tuỳ chọn, đỡ phải cài lại) |

> **Lưu ý về `paddlepaddle-gpu`:** bản trên PyPI **không kèm CUDA**. Phải cài từ
> index riêng của Paddle đúng phiên bản CUDA. Cell 1 tự dò qua `nvcc`.

---

## CELL 1 — Clone repo + cài dependency

```python
!git clone https://github.com/<user>/<repo>.git /kaggle/working/sinonom-ocr
%cd /kaggle/working/sinonom-ocr

# gỡ bản CPU Kaggle cài sẵn, thay bằng bản GPU đúng CUDA
import subprocess, re
nvcc = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True).stdout
m = re.search(r"release (\d+)\.(\d+)", nvcc)
major, minor = (int(m.group(1)), int(m.group(2))) if m else (12, 6)
tag = "cu130" if major >= 13 else ("cu129" if (major, minor) >= (12, 9) else ("cu126" if (major, minor) >= (12, 6) else "cu118"))
print("CUDA tag:", tag)

!pip uninstall -y paddlepaddle paddlepaddle-gpu -q
!pip install -q paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/{tag}/
!pip install -q "paddleocr>=3.7.0" scipy opencc-python-reimplemented opencv-python-headless
```

## CELL 2 — Kiểm tra GPU (dừng lại nếu fail)

```python
import paddle
print("paddle", paddle.__version__)
print("compiled_with_cuda:", paddle.device.is_compiled_with_cuda())
n_gpu = paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0
print("gpu_count:", n_gpu)
assert n_gpu > 0, "Paddle không thấy GPU — kiểm tra Accelerator và cell 1"
```

**Đừng bỏ qua cell này.** Nếu Paddle không thấy GPU nó vẫn chạy bằng CPU, không
báo lỗi gì, chỉ là chậm hơn ~10 lần và bạn phát hiện sau 1 tiếng.

## CELL 3 — Chạy thử 5 trang trước

```python
%cd /kaggle/working/sinonom-ocr
!PYTHONPATH=/kaggle/working/sinonom-ocr python scripts/run_han_ocr.py \
    --input dataset/china/q1 --limit 5 --device gpu:0 \
    --out /kaggle/working/q1_test.json
```

Xem log: mỗi trang nên ra ~20-30 cột, ~400-480 chữ, và **cần xem lại < 20%**.
Nếu số cột chỉ còn một chữ số hoặc chữ toàn 務/局/司 thì dừng, đừng chạy tiếp.

## CELL 4 — Chạy full, chia 2 GPU song song

```python
%cd /kaggle/working/sinonom-ocr
import subprocess, time
t0 = time.time()
ps = [subprocess.Popen(
        f"PYTHONPATH=/kaggle/working/sinonom-ocr python scripts/run_han_ocr.py "
        f"--input dataset/china/q1 --out /kaggle/working/q1.json "
        f"--device gpu:{i} --shard {i}/2", shell=True)
      for i in range(2)]
print("exit codes:", [p.wait() for p in ps])
print(f"{time.time()-t0:.0f}s")

!python scripts/merge_shards.py --out /kaggle/working/q1.json
```

Mỗi shard ghi file riêng (`q1.shard0.json`, `q1.shard1.json`), `merge_shards.py`
gộp lại theo đúng thứ tự trang rồi xoá shard.

## CELL 5 — Kiểm tra kết quả

```python
import json, re, collections
d = json.load(open('/kaggle/working/q1.json'))
CJK = re.compile(r'[一-鿿㐀-䶿]')
NOISE = set('務局司商創財員市房品號機業科貿發銀店濟')
rep = lambda t: 1 - len(set(t))/len(t) if t else 0
def bad(t):
    c = CJK.findall(t)
    return len(c) >= 4 and (sum(x in NOISE for x in c)/len(c) >= .30 or rep(t) >= .45)

print(f"{len({r['page'] for r in d})} trang, {len(d)} cột, {sum(len(r['text']) for r in d)} chữ")
print(f"consider=1 : {sum(r['consider'] for r in d)} ({100*sum(r['consider'] for r in d)/len(d):.1f}%)")
print(f"cột rác    : {sum(bad(r['text']) for r in d)} ({100*sum(bad(r['text']) for r in d)/len(d):.1f}%)")
print(f"giản thể sót: {sum(1 for r in d for ch in r['text'] if ch in set('务济国广见灵员业动华'))}")

# schema phải khớp đúng format cũ
keys = ('bbox','volume','page','page_number','text','middle','consider')
print("schema OK:", all(tuple(r.keys()) == keys for r in d))
```

**Ngưỡng tham chiếu** (đo trên 10 trang ở máy local, sau khi sửa):

| chỉ số | kỳ vọng |
|---|---|
| cột rác | < 2% |
| consider=1 | ~16% |
| giản thể sót | **0** |
| schema OK | True |

---

## Cách khác: chạy 1 lệnh

```python
!python scripts/kaggle_run_ocr.py --volume q1
```

Script tự cài, tự kiểm tra GPU, tự chia shard theo số GPU thấy được, tự gộp.
Thêm `--skip-install` nếu đã cài rồi, `--limit 5` để chạy thử.

---

## Về dataset

Repo đang chứa sẵn ảnh trong `dataset/china/q1/` (~76 ảnh). Nếu `git clone` quá
nặng hoặc ảnh không được push, upload thư mục ảnh thành **Kaggle Dataset** rồi trỏ vào:

```python
!python scripts/run_han_ocr.py --input /kaggle/input/<ten-dataset>/q1 \
    --out /kaggle/working/q1.json --device gpu:0
```

Ảnh `*_deleted.jpg` được loại tự động, không cần xoá tay.

## Nhớ tải kết quả về

File trong `/kaggle/working/` **mất khi session kết thúc**. Tải `q1.json` về, hoặc
Save Version để giữ lại output.

---

## Lỗi hay gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `compiled_with_cuda: False` | Cài nhầm bản CPU. Chạy lại cell 1, kiểm tra tag CUDA |
| `ModuleNotFoundError: core.interfaces` | Thiếu `core/interfaces.py` khi push. Kiểm tra file đã được commit |
| `ModuleNotFoundError: scipy` | `ocr_pipeline.py` import scipy ở module level — cài lại cell 1 |
| Tải model chậm/treo | Internet OFF, hoặc mạng Kaggle chậm. Model cache ở `~/.paddlex/official_models` |
| OOM trên GPU | Chạy 1 shard thay vì 2 (`--device gpu:0`, bỏ `--shard`) |
| Chạy chậm như CPU | Xem lại cell 2 — gần như chắc chắn Paddle không thấy GPU |

## Ước lượng thời gian

Local CPU (M-series) ~30s/trang → 71 trang ≈ 35 phút.
Kaggle T4 x1 kỳ vọng nhanh hơn đáng kể; T4 x2 chia đôi tiếp.
**Chưa đo thực tế trên Kaggle** — hãy xem log cell 3 để ước lượng trước khi chạy full.
