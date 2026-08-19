"""
Gộp các file shard của run_han_ocr.py thành một JSON duy nhất, sắp lại theo
đúng thứ tự trang.

    python scripts/merge_shards.py --out dataset/raw_han_ocr/q1.json
"""
import argparse
import glob
import json
import os
import re


def page_key(rec):
    """Sắp theo page_number, rồi theo tên file để giữ ổn định."""
    return (rec.get('page_number', 0), rec.get('page', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="File JSON đích, vd dataset/raw_han_ocr/q1.json")
    ap.add_argument("--keep-shards", action="store_true", help="Giữ lại file shard sau khi gộp")
    args = ap.parse_args()

    root, ext = os.path.splitext(args.out)
    shards = sorted(glob.glob(f"{root}.shard*{ext}"),
                    key=lambda p: int(re.search(r'\.shard(\d+)', p).group(1)))
    if not shards:
        print(f"Không tìm thấy shard nào khớp {root}.shard*{ext}")
        return 1

    records = []
    for s in shards:
        part = json.load(open(s, encoding='utf-8'))
        print(f"  {os.path.basename(s)}: {len(part)} cột")
        records.extend(part)

    # Trong mỗi trang, thứ tự cột do pipeline sinh ra đã đúng (phải->trái);
    # sort ổn định theo trang nên không phá thứ tự đó.
    records.sort(key=page_key)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    pages = len({r['page'] for r in records})
    chars = sum(len(r['text']) for r in records)
    print(f"\nĐã gộp {len(shards)} shard -> {args.out}")
    print(f"  {pages} trang, {len(records)} cột, {chars} chữ")

    if not args.keep_shards:
        for s in shards:
            os.remove(s)
        print(f"  đã xoá {len(shards)} file shard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
