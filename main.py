# NOTE: NOT REMOVE THIS LINE!
# pyrefly: ignore [missing-import]
import torch
import argparse
import os

from config import Config

def main():
    parser = argparse.ArgumentParser(description="Hán-Việt Parallel Corpus Builder")
    parser.add_argument("--han_dir", type=str, default=Config.HAN_DIR, help="Directory containing Hán input (images/text)")
    parser.add_argument("--viet_dir", type=str, default=Config.VIET_DIR, help="Directory containing Vietnamese text")
    parser.add_argument("--output_dir", type=str, default=Config.OUTPUT_DIR, help="Directory for output files")
    
    # Execution steps flags
    parser.add_argument("--first-n-images", type=int, default=None, help="Only OCR the first N images in each Hán folder")
    parser.add_argument("--do-ocr", action="store_true", help="Enable OCR processing")
    
    args = parser.parse_args()

    # ── XỬ LÝ CHÍNH TRỰC TIẾP TỪ MAIN ──
    if args.do_ocr:
        print("Starting OCR processing...")
        han_ocr_out_dir = os.path.join(args.output_dir, "han_ocr")
        os.makedirs(han_ocr_out_dir, exist_ok=True)
        
        if not os.path.exists(args.han_dir):
            print(f"Error: Directory {args.han_dir} does not exist.")
            return

        han_works = sorted([d for d in os.listdir(args.han_dir) if os.path.isdir(os.path.join(args.han_dir, d))])
        if not han_works:
            print(f"No work directories found in {args.han_dir}")
            return
            
        import json
        from ocr.ocr_pipeline import ocr_sinonom_page
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        
        print("Initializing global PaddleOCR engine for batch processing...")
        ocr_engine = PaddleOCR(lang='chinese_cht', use_textline_orientation=True)

        for work_id in han_works:
            han_input_path = os.path.join(args.han_dir, work_id)
            
            # Output là JSON 100%
            output_path = os.path.join(han_ocr_out_dir, f"{work_id}.json")
            
            images = sorted([f for f in os.listdir(han_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if args.first_n_images is not None:
                images = images[:args.first_n_images]
                
            work_results = []
            for img_name in images:
                # 1. Bỏ qua các ảnh bị xóa
                if "_deleted" in img_name.lower():
                    print(f"Skipping deleted image: {img_name}")
                    continue

                print(f"Processing image: {img_name}")
                img_path = os.path.join(han_input_path, img_name)
                try:
                    # 2. Xử lý OCR qua pipeline chuẩn công nghiệp
                    page_results = ocr_sinonom_page(
                        image_path=img_path, 
                        debug_ocr=False, 
                        output_dir=None,
                        ocr_engine=ocr_engine
                    )
                    
                    if isinstance(page_results, list):
                        work_results.extend(page_results)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
            # 3. Lưu file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(work_results, f, ensure_ascii=False, indent=4)
                
            print(f"Saved OCR output to {output_path}")

    else:
        print("No operation specified. Use --do-ocr to run OCR processing.")

    print("Done!")

if __name__ == "__main__":
    main()
