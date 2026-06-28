# Sino-Nom OCR & Parallel Corpus Builder

This tool automates the extraction and alignment of historical Sino-Nom (Traditional Chinese / Hán Nôm) manuscripts with their modern Vietnamese translations. It features an optimized OCR pipeline (via PaddleOCR) for vertical text, PDF extraction, NLP segmentation, and sentence-level alignment.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run the full pipeline or specific steps using `main.py`. The system maps inputs by exact name matching (e.g., folder `dataset/china/q1` is automatically paired with PDF `dataset/vietnam/q1.pdf`).

### Basic Command (Run everything)
By default, if no steps are specified, the script runs the entire pipeline (`--run_all`):
```bash
python3 main.py --han_dir dataset/china --viet_dir dataset/vietnam --output_dir output
```

### Optional Command-Line Arguments

You can customize directories or run specific steps of the pipeline using the following optional flags:

#### Directory Configuration
- `--han_dir` : Path to the directory containing Hán input folders (default: `dataset/china`).
- `--viet_dir` : Path to the directory containing Vietnamese PDF translations (default: `dataset/vietnam`).
- `--output_dir` : Path where the final aligned TSV and Excel files will be saved (default: `output`).

#### Execution Steps (Pipelines)
Use these flags to run only specific parts of the process, which is useful for debugging or resuming work:
- `--step_ocr` : Run **Step 1** only. Performs OCR on the Hán images and saves the raw text to `han_raw.txt`.
- `--step_seg` : Run **Step 2** only. Reads the translated PDF, and runs NLP segmentation (Underthesea) on both the Hán and Việt texts.
- `--step_align` : Run **Step 3** only. Runs the alignment algorithm (TF-IDF Cosine Similarity / BERT) to map Hán sentences to Việt sentences (m-n mapping) and exports the Excel/TSV files.
- `--run_all` : Explicitly run all the steps above sequentially.
- `--first-n-images` : (Optional) Limit OCR to only the first *N* images in each Hán input folder. Useful for testing pipelines on large books. Usage: `--first-n-images 5`

### Example: Running only the Alignment step
If you have already run OCR and Segmentation and just want to re-run the alignment with different configurations:
```bash
python3 main.py --step_align
```

## Configuration (.env)
You can set default API keys and paths in a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_api_key_here"
```
*(Note: API keys are strictly optional. The system will gracefully fall back to local models like PaddleOCR and BERTAlign if no keys are provided.)*
