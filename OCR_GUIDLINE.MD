# OCR Guideline for Sino-Nom Historical Documents

This document outlines the best practices and the technical pipeline used for extracting Sino-Nom (Traditional Chinese / Hán Nôm) characters from historical manuscripts, specifically optimizing for vertical reading order and paper deterioration.

## 1. Provider Selection
- **Primary OCR Engine**: `PaddleOCR` (Local, Free, highly robust for Traditional Chinese).
- **Why not Cloud APIs?**: 
  - Google Cloud Vision requires billing enabled and can be costly for large datasets.
  - LLMs (like Gemini) can struggle with strict coordinate-based layout retention for dense numerical tables and are restricted by API quotas.

## 2. Image Preprocessing (Minimalist Approach)
Deep learning OCR models (like PaddleOCR's CRNN) are trained on raw textures, including lighting variations and noise. Extreme binarization often does more harm than good.
- **White Padding**: Added a 50px white border around the image using `cv2.copyMakeBorder`. This is crucial to prevent characters written near the physical edge of the paper (e.g., "畝" at the bottom) from being cropped out during CNN feature extraction.
- **Grayscale Conversion**: Converted to grayscale (`cv2.COLOR_BGR2GRAY`) to remove color noise from yellowed paper and seals, but retained the soft anti-aliased edges of the ink strokes.
- **Avoided Morphological/Thresholding**: Deliberately removed Adaptive Thresholding and Morphological Closing. While these can theoretically connect broken strokes (e.g., fixing 千 vs 十), they frequently introduce severe noise by turning paper wrinkles, water stains, and red seals into "hallucinated" characters.

## 3. Layout Correction & Reading Order
Historical Sino-Nom documents are read **Top-to-Bottom** and **Right-to-Left**. Standard OCR engines default to Left-to-Right reading.
- **90-Degree CCW Rotation**: The image is rotated 90 degrees counter-clockwise before inference. This converts vertical columns into horizontal rows.
- **Band-based Bounding Box Sorting**:
  1. Extracted bounding boxes are sorted into "bands" based on their `center_x` coordinates (which correspond to the original vertical `y` positions).
  2. Bands are sorted in descending order so that the bottom sections (often containing structural totals like 總計) appear first.
  3. Within each band, boxes are sorted by their `center_y` coordinates ascending (which corresponds to the original horizontal `x` positions, ensuring the right-most columns are read first).

## 4. PaddleOCR Hyperparameter Tuning
- `lang='chinese_cht'`: Forced Traditional Chinese character set.
- `use_angle_cls=True`: Enabled angle classifier to fix minor tilts.
- `det_db_unclip_ratio=1.8`: Increased from the default (1.5) to expand the detected text boxes slightly, preventing the edges of characters from being sliced off.
- `det_db_box_thresh=0.4`: Lowered from the default (0.6) to increase sensitivity, ensuring faint or thin strokes (like the character 銀 - Silver) are not discarded as background noise.
- **Confidence Filter**: Post-processing ignores any detected box with a confidence score `< 0.4` to filter out residual noise.

## 5. Post-Correction Strategy
Instead of distorting the image with aggressive OpenCV morphology (which damages character integrity), inevitable OCR misidentifications are handled via post-correction regex/dictionaries based on contextual domain knowledge:
- `十五萬` $\rightarrow$ `丁五萬` (Confusing 'Thập' with 'Đinh' in population counts).
- `部` $\rightarrow$ `畝` (Confusing 'Bộ' with 'Mẫu' in land measurements).
- `丁八十` $\rightarrow$ `丁八千` (Confusing 'Thập' with 'Thiên' due to broken top strokes).

---
*Generated during the Sino-Nom OCR optimization phase.*
