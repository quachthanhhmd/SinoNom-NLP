"""
Chạy OCR PaddleOCR trên một thư mục ảnh Hán Nôm và xuất JSON theo đúng định dạng
của dataset/raw_han_ocr/<volume>.json:

    [{"bbox", "volume", "page", "page_number", "text", "middle", "consider"}, ...]

Ví dụ:
    python scripts/run_han_ocr.py --input dataset/china/q1 --out dataset/raw_han_ocr/q1.json
    python scripts/run_han_ocr.py --input dataset/china/q1 --limit 3   # thử vài trang trước
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.ocr_pipeline import ocr_sinonom_page, build_ocr_engine

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_pages(input_dir: str, include_deleted: bool = False):
    """Ảnh có hậu tố _deleted là trang đã bị loại khỏi corpus, mặc định bỏ qua."""
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS)]
    if not include_deleted:
        files = [f for f in files if "_deleted" not in f]
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="Sino-Nom OCR -> raw_han_ocr JSON")
    ap.add_argument("--input", required=True, help="Thư mục ảnh, vd dataset/china/q1")
    ap.add_argument("--out", default=None, help="File JSON đích")
    ap.add_argument("--limit", type=int, default=None, help="Chỉ chạy N trang đầu")
    ap.add_argument("--pages", default=None, help="Danh sách tên file, cách nhau bởi dấu phẩy")
    ap.add_argument("--include-deleted", action="store_true", help="Gồm cả ảnh *_deleted")
    ap.add_argument("--debug-ocr", action="store_true", help="Bọc [chữ?] cho ký tự độ tin cậy thấp")
    ap.add_argument("--device", default=None,
                    help="'gpu:0', 'gpu:1', 'cpu'. Bỏ trống = tự dò")
    ap.add_argument("--shard", default=None,
                    help="Chia việc cho nhiều tiến trình, dạng I/N (vd 0/2). Mỗi shard ghi 1 file riêng.")
    args = ap.parse_args()

    volume = os.path.basename(os.path.normpath(args.input))
    out_path = args.out or os.path.join("dataset", "raw_han_ocr", f"{volume}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pages = list_pages(args.input, args.include_deleted)
    if args.pages:
        wanted = {p.strip() for p in args.pages.split(",")}
        pages = [p for p in pages if p in wanted]
    if args.limit:
        pages = pages[: args.limit]

    if args.shard:
        idx, total = (int(x) for x in args.shard.split('/'))
        pages = pages[idx::total]
        root, ext = os.path.splitext(out_path)
        out_path = f"{root}.shard{idx}{ext}"

    if not pages:
        print(f"Không tìm thấy ảnh nào trong {args.input}")
        return 1

    print(f"Volume '{volume}': {len(pages)} trang -> {out_path} (device={args.device or 'auto'})")
    engine = build_ocr_engine(device=args.device)

    records = []
    t0 = time.time()
    for i, name in enumerate(pages, 1):
        path = os.path.join(args.input, name)
        started = time.time()
        try:
            page_records = ocr_sinonom_page(
                image_path=path,
                ocr_engine=engine,
                debug_ocr=args.debug_ocr,
            )
        except Exception as e:
            print(f"  [{i}/{len(pages)}] {name}: LỖI {type(e).__name__}: {e}")
            continue

        records.extend(page_records)
        chars = sum(len(r["text"]) for r in page_records)
        flagged = sum(r["consider"] for r in page_records)
        print(
            f"  [{i}/{len(pages)}] {name}: {len(page_records):3d} cột, "
            f"{chars:5d} chữ, {flagged:3d} cần xem lại ({time.time()-started:.1f}s)"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    total_chars = sum(len(r["text"]) for r in records)
    total_flagged = sum(r["consider"] for r in records)
    print(
        f"\nXong {len(pages)} trang trong {time.time()-t0:.0f}s: "
        f"{len(records)} cột, {total_chars} chữ, "
        f"{total_flagged} cột cần xem lại ({100*total_flagged/max(1,len(records)):.1f}%)"
    )
    print(f"Đã ghi: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
