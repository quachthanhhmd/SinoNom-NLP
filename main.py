# NOTE: NOT REMOVE THIS LINE!
# pyrefly: ignore [missing-import]
import argparse
import os

# Tắt MKLDNN (OneDNN) để né lỗi ConvertPirAttribute2RuntimeAttribute của PaddlePaddle 3.x trên CPU
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_ENABLE_MKLDNN"] = "0"

from config import Config

def main():
    parser = argparse.ArgumentParser(description="Hán-Việt Parallel Corpus Builder")
    parser.add_argument("--han_dir", type=str, default=Config.HAN_DIR, help="Directory containing Hán input (images/text)")
    parser.add_argument("--viet_dir", type=str, default=Config.VIET_DIR, help="Directory containing Vietnamese text")
    parser.add_argument("--output_dir", type=str, default=Config.OUTPUT_DIR, help="Directory for output files")
    
    # Execution steps flags
    parser.add_argument("--first-n-images", type=int, default=None, help="Only OCR the first N images in each Hán folder")
    parser.add_argument("--do-ocr", action="store_true", help="Enable OCR processing")
    parser.add_argument("--except-folders", nargs='+', default=[], help="List of folder names to exclude from OCR (e.g. --except-folders q1 q2)")
    
    # PDF & Config flags
    parser.add_argument("--han_pdf_dir", type=str, default="dataset/china/han_pdf", help="Directory containing Hán PDF files")
    parser.add_argument("--run-config", type=str, default="run_config.json", help="Path to JSON run configuration")
    parser.add_argument("--ocr-pdf", type=str, default=None, help="Specific PDF file to OCR (e.g. 02.pdf). Skips all other files.")
    
    args = parser.parse_args()

    # ── XỬ LÝ CHÍNH TRỰC TIẾP TỪ MAIN ──
    if args.do_ocr:
        print("Starting OCR processing...")
        han_ocr_out_dir = os.path.join(args.output_dir, "han_ocr")
        os.makedirs(han_ocr_out_dir, exist_ok=True)
        import json
        import numpy as np
        
        # Load run config
        run_config = {}
        if os.path.exists(args.run_config):
            try:
                with open(args.run_config, 'r', encoding='utf-8') as f:
                    run_config = json.load(f)
                print(f"Loaded configuration from {args.run_config}")
            except Exception as e:
                print(f"Failed to load {args.run_config}: {e}")

        from ocr.ocr_pipeline import ocr_sinonom_page
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        
        print("Initializing global PaddleOCR engine for batch processing (CPU mode)...")
        ocr_engine = PaddleOCR(lang='chinese_cht', use_textline_orientation=True)

        # ---------------------------------------------------------
        # 1. XỬ LÝ ẢNH TRONG CÁC THƯ MỤC (IMAGE FOLDERS)
        # ---------------------------------------------------------
        if os.path.exists(args.han_dir) and not args.ocr_pdf:
            han_works = sorted([d for d in os.listdir(args.han_dir) if os.path.isdir(os.path.join(args.han_dir, d))])
            if args.except_folders:
                han_works = [d for d in han_works if d not in args.except_folders]
                
            for work_id in han_works:
                work_cfg = run_config.get(work_id, {})
                if work_cfg.get("exclude", False):
                    print(f"Skipping directory {work_id} due to config exclude.")
                    continue
                    
                is_half_page = work_cfg.get("is_half_page", False)
                skip_pages = work_cfg.get("skip_pages", [])
                
                han_input_path = os.path.join(args.han_dir, work_id)
                output_path = os.path.join(han_ocr_out_dir, f"{work_id}.json")
                
                images = sorted([f for f in os.listdir(han_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if args.first_n_images is not None:
                    images = images[:args.first_n_images]
                    
                work_results = []
                for img_name in images:
                    if "_deleted" in img_name.lower():
                        continue
                        
                    # Trích xuất số trang từ tên file để check skip_pages
                    import re
                    m = re.search(r'(\d+)', img_name)
                    page_num = int(m.group(1)) if m else -1
                    if page_num in skip_pages:
                        print(f"Skipping {img_name} (in skip_pages)")
                        continue

                    print(f"Processing image: {img_name}")
                    img_path = os.path.join(han_input_path, img_name)
                    try:
                        page_results = ocr_sinonom_page(
                            image_path=img_path, 
                            debug_ocr=False, 
                            output_dir=None,
                            ocr_engine=ocr_engine,
                            is_half_page=is_half_page
                        )
                        if isinstance(page_results, list):
                            work_results.extend(page_results)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(work_results, f, ensure_ascii=False, indent=4)
                print(f"Saved OCR output to {output_path}")

        # ---------------------------------------------------------
        # 2. XỬ LÝ CÁC FILE PDF (PDF FILES)
        # ---------------------------------------------------------
        if os.path.exists(args.han_pdf_dir):
            try:
                import fitz # PyMuPDF
                import cv2
            except ImportError:
                print("PyMuPDF (fitz) is not installed. Skipping PDF processing.")
                fitz = None
                
            if fitz:
                pdf_files = sorted([f for f in os.listdir(args.han_pdf_dir) if f.lower().endswith('.pdf')])
                if args.ocr_pdf:
                    pdf_files = [f for f in pdf_files if f == args.ocr_pdf]
                    
                for pdf_name in pdf_files:
                    work_id = os.path.splitext(pdf_name)[0]
                    work_cfg = run_config.get(work_id, {})
                    if work_cfg.get("exclude", False):
                        print(f"Skipping PDF {pdf_name} due to config exclude.")
                        continue
                        
                    is_half_page = work_cfg.get("is_half_page", False)
                    skip_pages = work_cfg.get("skip_pages", [])
                    
                    pdf_path = os.path.join(args.han_pdf_dir, pdf_name)
                    output_path = os.path.join(han_ocr_out_dir, f"{work_id}_pdf.json")
                    
                    print(f"Opening PDF: {pdf_name}")
                    try:
                        doc = fitz.open(pdf_path)
                    except Exception as e:
                        print(f"Failed to open PDF {pdf_name}: {e}")
                        continue
                        
                    work_results = []
                    num_pages = len(doc)
                    processed_count = 0
                    
                    for page_index in range(num_pages):
                        if args.first_n_images is not None and processed_count >= args.first_n_images:
                            break
                            
                        page_num_1_indexed = page_index + 1
                        if page_num_1_indexed in skip_pages:
                            print(f"Skipping PDF {pdf_name} page {page_num_1_indexed} (in skip_pages)")
                            continue
                            
                        print(f"Processing PDF {pdf_name}, page {page_num_1_indexed}/{num_pages}")
                        processed_count += 1
                        
                        try:
                            page = doc[page_index]
                            # Render to high-res image (300 DPI equivalent)
                            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                            
                            # Xử lý RGB to BGR nếu cần (OpenCV dùng BGR)
                            if pix.n == 3:
                                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                            elif pix.n == 4:
                                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                                
                            debug_output_dir = None
                            if args.ocr_pdf:
                                debug_output_dir = os.path.join(han_ocr_out_dir, f"debug_{work_id}_page_{page_num_1_indexed}")
                                
                            page_results = ocr_sinonom_page(
                                image_path=None,
                                image_np=img_np,
                                debug_ocr=False,
                                output_dir=debug_output_dir,
                                ocr_engine=ocr_engine,
                                is_half_page=is_half_page,
                                pdf_page_num=page_num_1_indexed,
                                pdf_filename=pdf_name
                            )
                            if isinstance(page_results, list):
                                work_results.extend(page_results)
                        except Exception as e:
                            print(f"Error processing PDF {pdf_name} page {page_num_1_indexed}: {e}")
                            
                    doc.close()
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(work_results, f, ensure_ascii=False, indent=4)
                    print(f"Saved PDF OCR output to {output_path}")

    else:
        print("No operation specified. Use --do-ocr to run OCR processing.")

    print("Done!")

if __name__ == "__main__":
    main()
