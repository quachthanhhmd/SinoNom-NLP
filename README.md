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
- `--step_align` : Run **Step 3** only. Runs the semantic ensemble and constrained monotonic m-n decoder, then exports Excel/TSV files.
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
*(API keys are optional for OCR, Phase 1 and Phase 2. Phase 3 uses Gemini when `GEMINI_API_KEY` is available and otherwise falls back to the local Qwen model; Gemini is recommended for recovering more exact beads.)*

For the checked-in `dataset/MAPPING` corpus, run:

```bash
python run_mapping.py --aligner ensemble --qwen --realign --repair-rounds 3
```

Phase 2 checks every two-sided bead using the strict `exact/addition/omission/mismatch` rubric. Phase 3 restores rejected merges to atomic source rows and iteratively redraws local m-n boundaries; exact beads become immutable anchors. The existing fingerprinted Phase-1 embedding cache is reused.

Only `*_exact_accepted.tsv` is consumed by `scripts/prepare_data.py`. Diagnostic partitions include `*_addition.tsv`, `*_omission.tsv`, `*_mismatch.tsv`, `*_review.tsv`, `*_unmatched.tsv`, an evaluation report, and a deterministic independent-review sample workbook. Existing `HVB_001_parallel.*` files remain audit baselines until the new pipeline is rerun.
