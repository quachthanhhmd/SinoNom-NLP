import os
from core.interfaces import OCRProvider, OCRCorrector, Segmenter, Aligner
from utils.exporters import CorpusExporter

class CorpusPipeline:
    def __init__(
        self,
        ocr_provider: OCRProvider,
        ocr_corrector: OCRCorrector,
        han_segmenter: Segmenter,
        viet_segmenter: Segmenter,
        aligner: Aligner,
        exporter: CorpusExporter
    ):
        self.ocr = ocr_provider
        self.corrector = ocr_corrector
        self.han_segmenter = han_segmenter
        self.viet_segmenter = viet_segmenter
        self.aligner = aligner
        self.exporter = exporter

    def process_work(self, work_id: str, han_input: str, viet_input: str, run_ocr: bool = True, run_seg: bool = True, run_align: bool = True, first_n_images: int = None):
        """
        Process a single work.
        han_input is a directory path containing multiple Hán images.
        viet_input is a PDF file path.
        """
        print(f"--- Processing Work: {work_id} ---")
        
        han_raw_file = os.path.join(self.exporter.output_dir, f"{work_id}_han_raw.txt")
        viet_raw_file = os.path.join(self.exporter.output_dir, f"{work_id}_viet_raw.txt")
        han_sent_file = os.path.join(self.exporter.output_dir, f"{work_id}_han_sentences.txt")
        viet_sent_file = os.path.join(self.exporter.output_dir, f"{work_id}_viet_sentences.txt")
        
        # 1. Obtain Hán Text (from directory of images)
        if run_ocr:
            print("[Step 1] Running Hán OCR...")
            han_texts = []
            if os.path.isdir(han_input):
                # Sort files to ensure page order
                image_files = sorted([
                    f for f in os.listdir(han_input) 
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                
                if first_n_images is not None:
                    image_files = image_files[:first_n_images]
                
                for img_file in image_files:
                    img_path = os.path.join(han_input, img_file)
                    raw_han_page = self.ocr.extract_text(img_path)
                    
                    if self.corrector:
                        corrected_page = self.corrector.correct(raw_han_page)
                        han_texts.append(corrected_page)
                    else:
                        han_texts.append(raw_han_page)
                        
                han_text = "\n".join(han_texts)
                with open(han_raw_file, 'w', encoding='utf-8') as f:
                    f.write(han_text)
            else:
                print(f"Error: {han_input} is not a directory.")
                return
        
        # 2. Text Segmentation
        if run_seg:
            print("[Step 2] Extracting PDF and Segmenting Texts...")
            # Load Hán Text
            if os.path.exists(han_raw_file):
                with open(han_raw_file, 'r', encoding='utf-8') as f:
                    han_text = f.read()
            else:
                print(f"Warning: {han_raw_file} not found. Cannot segment Hán text.")
                han_text = ""
                
            # Extract Viet Text from PDF
            viet_text = self._extract_text_from_pdf(viet_input)
            with open(viet_raw_file, 'w', encoding='utf-8') as f:
                f.write(viet_text)

            # Segment
            han_sentences = self.han_segmenter.segment(han_text)
            viet_sentences = self.viet_segmenter.segment(viet_text)
            
            with open(han_sent_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(han_sentences))
            with open(viet_sent_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(viet_sentences))

        # 3. Alignment
        if run_align:
            print("[Step 3] Running Alignment...")
            if os.path.exists(han_sent_file) and os.path.exists(viet_sent_file):
                with open(han_sent_file, 'r', encoding='utf-8') as f:
                    han_sentences = [line.strip() for line in f if line.strip()]
                with open(viet_sent_file, 'r', encoding='utf-8') as f:
                    viet_sentences = [line.strip() for line in f if line.strip()]
                
                aligned_data = self.aligner.align(han_sentences, viet_sentences)
                self.exporter.export_parallel_tsv(work_id, aligned_data)
                self.exporter.export_parallel_excel(work_id, aligned_data)
            else:
                print("Error: Missing sentence files for alignment. Run --step_seg first.")

    def process_global(self, han_inputs: list, viet_inputs: list, run_ocr: bool = True, run_seg: bool = True, run_align: bool = True, first_n_images: int = None):
        """
        Process all Hán inputs and all Việt inputs globally (concatenating everything).
        """
        work_id = "global"
        print(f"--- Processing Global Work ---")
        
        han_raw_file = os.path.join(self.exporter.output_dir, f"{work_id}_han_raw.txt")
        viet_raw_file = os.path.join(self.exporter.output_dir, f"{work_id}_viet_raw.txt")
        han_sent_file = os.path.join(self.exporter.output_dir, f"{work_id}_han_sentences.txt")
        viet_sent_file = os.path.join(self.exporter.output_dir, f"{work_id}_viet_sentences.txt")
        
        # 1. Obtain Hán Text (from multiple directories)
        if run_ocr:
            print("[Step 1] Running Global Hán OCR...")
            global_han_text = ""
            for han_input in han_inputs:
                print(f"  Extracting from: {han_input}")
                han_texts = []
                if os.path.isdir(han_input):
                    image_files = sorted([f for f in os.listdir(han_input) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                    if first_n_images is not None:
                        image_files = image_files[:first_n_images]
                    
                    for img_file in image_files:
                        img_path = os.path.join(han_input, img_file)
                        raw_han_page = self.ocr.extract_text(img_path)
                        if self.corrector:
                            han_texts.append(self.corrector.correct(raw_han_page))
                        else:
                            han_texts.append(raw_han_page)
                            
                    global_han_text += "\n".join(han_texts) + "\n"
                else:
                    print(f"  Error: {han_input} is not a directory. Skipping.")
            
            with open(han_raw_file, 'w', encoding='utf-8') as f:
                f.write(global_han_text)
                
        # 2. Text Segmentation
        if run_seg:
            print("[Step 2] Extracting PDF and Segmenting Texts...")
            if os.path.exists(han_raw_file):
                with open(han_raw_file, 'r', encoding='utf-8') as f:
                    han_text = f.read()
            else:
                print(f"Warning: {han_raw_file} not found.")
                han_text = ""
                
            global_viet_text = ""
            for viet_input in viet_inputs:
                print(f"  Extracting from PDF: {viet_input}")
                viet_text = self._extract_text_from_pdf(viet_input)
                global_viet_text += viet_text + "\n"
                
            with open(viet_raw_file, 'w', encoding='utf-8') as f:
                f.write(global_viet_text)

            han_sentences = self.han_segmenter.segment(han_text)
            viet_sentences = self.viet_segmenter.segment(global_viet_text)
            
            with open(han_sent_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(han_sentences))
            with open(viet_sent_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(viet_sentences))

        # 3. Alignment
        if run_align:
            print("[Step 3] Running Global Alignment...")
            if os.path.exists(han_sent_file) and os.path.exists(viet_sent_file):
                with open(han_sent_file, 'r', encoding='utf-8') as f:
                    han_sentences = [line.strip() for line in f if line.strip()]
                with open(viet_sent_file, 'r', encoding='utf-8') as f:
                    viet_sentences = [line.strip() for line in f if line.strip()]
                
                aligned_data = self.aligner.align(han_sentences, viet_sentences)
                self.exporter.export_parallel_tsv(work_id, aligned_data)
                self.exporter.export_parallel_excel(work_id, aligned_data)
            else:
                print("Error: Missing sentence files for alignment. Run --step_seg first.")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Helper method to extract text from PDF."""
        # Using PyPDF2 / pypdf as a typical approach.
        # If fitz (PyMuPDF) is preferred, you can replace this logic.
        try:
            import PyPDF2
            text = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except ImportError:
            print("PyPDF2 is not installed. Please install it to extract Vietnamese PDFs: pip install PyPDF2")
            print(f"[Mock PDF Extraction] from {pdf_path}")
            return f"Mock content extracted from {os.path.basename(pdf_path)}"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""

