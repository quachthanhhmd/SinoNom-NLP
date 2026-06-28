import argparse
import os

from config import Config
from ocr.providers import KanDianGuJiOCR, PaddleOCRProvider
from ocr.ensemble import EnsembleOCR
from ocr.corrector import GeminiOCRCorrector
from nlp.segmenter import RegexSegmenter, UndertheseaSegmenter
from nlp.aligner import BERTAlignerWrapper, TranslationCosineAligner
from core.pipeline import CorpusPipeline
from utils.exporters import CorpusExporter

def main():
    parser = argparse.ArgumentParser(description="Hán-Việt Parallel Corpus Builder")
    parser.add_argument("--han_dir", type=str, default=Config.HAN_DIR, help="Directory containing Hán input (images/text)")
    parser.add_argument("--viet_dir", type=str, default=Config.VIET_DIR, help="Directory containing Vietnamese text")
    parser.add_argument("--output_dir", type=str, default=Config.OUTPUT_DIR, help="Directory for output files")
    
    # Execution steps flags
    parser.add_argument("--step_ocr", action="store_true", help="Run Step 1: Hán OCR and save to han_raw.txt")
    parser.add_argument("--step_seg", action="store_true", help="Run Step 2: Read PDF and Segment both Hán and Việt")
    parser.add_argument("--step_align", action="store_true", help="Run Step 3: Alignment (m-n mapping)")
    parser.add_argument("--run_all", action="store_true", help="Run all steps sequentially")
    parser.add_argument("--first-n-images", type=int, default=None, help="Only OCR the first N images in each Hán folder")
    parser.add_argument("--global-concat", action="store_true", help="Merge all Hán folders and all Việt PDFs globally before processing")
    
    args = parser.parse_args()
    
    # If no specific step is requested, we can either prompt or run all. Let's default to run_all if nothing is set.
    if not (args.step_ocr or args.step_seg or args.step_align or args.run_all):
        print("No steps specified. Defaulting to --run_all.")
        args.run_all = True
        
    if args.run_all:
        args.step_ocr = True
        args.step_seg = True
        args.step_align = True

    print("Initializing components...")
    
    # 1. Init OCR providers
    print("Initializing OCR providers...")
    gemini_key = Config.GEMINI_API_KEY
    is_gemini_valid = gemini_key and gemini_key != ""
    
    kandian_token = getattr(Config, 'KANDIAN_TOKEN', '')
    kandian_email = getattr(Config, 'KANDIAN_EMAIL', '')
    
    providers = []
    
    if kandian_token and kandian_email:
        providers.append(KanDianGuJiOCR(token=kandian_token, email=kandian_email))
        
    providers.append(PaddleOCRProvider())
        
    ocr_ensemble = EnsembleOCR(providers=providers)
    
    # 2. OCR Corrector
    corrector = None  # Disabled GeminiOCRCorrector
    
    # 3. NLP Components
    han_segmenter = RegexSegmenter(lang="han")
    viet_segmenter = UndertheseaSegmenter()
    aligner = BERTAlignerWrapper()  # Default to BERT since Gemini is disabled
    
    # 4. Exporter
    exporter = CorpusExporter(output_dir=args.output_dir)
    
    # 5. Pipeline
    pipeline = CorpusPipeline(
        ocr_provider=ocr_ensemble,
        ocr_corrector=corrector,
        han_segmenter=han_segmenter,
        viet_segmenter=viet_segmenter,
        aligner=aligner,
        exporter=exporter
    )

    # Process files
    if not os.path.exists(args.han_dir):
        print(f"Error: Directory {args.han_dir} does not exist.")
        return
    if not os.path.exists(args.viet_dir):
        print(f"Error: Directory {args.viet_dir} does not exist.")
        return

    han_works = sorted([d for d in os.listdir(args.han_dir) if os.path.isdir(os.path.join(args.han_dir, d))])
    viet_pdfs = sorted([f for f in os.listdir(args.viet_dir) if f.lower().endswith(".pdf")])
    
    if not han_works:
        print(f"No work directories found in {args.han_dir}")
        return

    if args.global_concat:
        han_input_paths = [os.path.join(args.han_dir, w) for w in han_works]
        viet_input_paths = [os.path.join(args.viet_dir, p) for p in viet_pdfs]
        print(f"Global Concatenation Mode: Combining {len(han_input_paths)} Hán directories and {len(viet_input_paths)} Việt PDFs.")
        
        try:
            pipeline.process_global(
                han_inputs=han_input_paths,
                viet_inputs=viet_input_paths,
                run_ocr=args.step_ocr,
                run_seg=args.step_seg,
                run_align=args.step_align,
                first_n_images=args.first_n_images
            )
        except Exception as e:
            print(f"Error processing globally: {e}")
            
    else:
        for work_id in han_works:
            han_input_path = os.path.join(args.han_dir, work_id)
            expected_pdf = f"{work_id}.pdf"
            viet_input_path = os.path.join(args.viet_dir, expected_pdf)
            
            if not os.path.exists(viet_input_path):
                print(f"Warning: Corresponding PDF '{expected_pdf}' not found in {args.viet_dir} for folder '{work_id}'. Skipping.")
                continue
                
            print(f"Mapping Hán directory '{work_id}' to Việt PDF '{expected_pdf}'")
            
            try:
                pipeline.process_work(
                    work_id=work_id,
                    han_input=han_input_path,
                    viet_input=viet_input_path,
                    run_ocr=args.step_ocr,
                    run_seg=args.step_seg,
                    run_align=args.step_align,
                    first_n_images=args.first_n_images
                )
            except Exception as e:
                print(f"Error processing {work_id}: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
