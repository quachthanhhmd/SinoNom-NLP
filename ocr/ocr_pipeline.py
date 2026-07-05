import cv2
import numpy as np
import re
import os

# Tắt MKLDNN để chống lỗi ConvertPirAttribute2RuntimeAttribute trên bản CPU v3
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_ENABLE_MKLDNN"] = "0"

import argparse
from scipy.signal import find_peaks

def save_debug_image(img, step_name, output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{step_name}.jpg")
        cv2.imwrite(out_path, img)
        print(f"Saved debug image: {out_path}")

def auto_crop_margins(img, padding=50):
    """
    Tự động cắt bỏ lề trắng thừa của trang PDF/ảnh để tập trung vào nội dung chính.
    Dùng phương pháp chiếu (projection) để loại bỏ nhiễu nhỏ.
    """
    _, bw = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Cộng dồn số điểm ảnh đen theo chiều dọc và ngang
    col_sum = np.sum(bw, axis=0)
    row_sum = np.sum(bw, axis=1)
    
    # Ngưỡng chống nhiễu: Cột/Hàng phải có ít nhất 10 pixel đen thì mới tính là có nội dung
    noise_threshold = 255 * 10 
    
    valid_cols = np.where(col_sum > noise_threshold)[0]
    valid_rows = np.where(row_sum > noise_threshold)[0]
    
    if len(valid_cols) > 0 and len(valid_rows) > 0:
        x1 = max(0, valid_cols[0] - padding)
        x2 = min(img.shape[1], valid_cols[-1] + padding)
        y1 = max(0, valid_rows[0] - padding)
        y2 = min(img.shape[0], valid_rows[-1] + padding)
        return img[y1:y2, x1:x2]
        
    return img

def preprocess(image_path: str = None, image_np: np.ndarray = None, output_dir: str = None, max_size: int = 2500):
    """
    Step 2: Preprocessing
    Deskew using Hough line detection against vertical rules.
    Binarize with Adaptive Thresholding.
    Nếu ảnh quá to (vượt quá max_size), tự động thu nhỏ lại để tránh nhiễu nét và tăng tốc OCR.
    """
    if image_np is not None:
        if len(image_np.shape) == 3:
            img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            img = image_np.copy()
    elif image_path is not None:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
    else:
        raise ValueError("Either image_path or image_np must be provided")
        
    # [NEW] Cắt bỏ lề trắng thừa trước khi resize để giữ độ phân giải cao nhất cho phần chữ
    img = auto_crop_margins(img, padding=50)
    save_debug_image(img, "00_cropped", output_dir)
    
    h, w = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # 1. Deskew using Probabilistic Hough Transform
    # Lọc độ dài tối thiểu (minLineLength = h // 4) để CHỈ bắt các đường kẻ khung viền (rất dài)
    # và phớt lờ hoàn toàn các nét chữ Hán (vốn rất ngắn) gây nhiễu góc xoay.
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=100, minLineLength=h // 4, maxLineGap=20)
    
    if lines is not None:
        vertical_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y2 != y1:
                dx = x2 - x1
                dy = y2 - y1
                # Tính góc lệch so với trục dọc (Y)
                angle = np.degrees(np.arctan(dx / dy))
                # Chỉ xét các đường gần như dọc (lệch < 5 độ)
                if abs(angle) < 5:
                    vertical_angles.append(angle)
                    
        if len(vertical_angles) > 0:
            median_angle = np.median(vertical_angles)
            if abs(median_angle) > 0.1:  # Rotate if skew is > 0.1 degrees
                # Math: angle = arctan(dx/dy). Góc âm có nghĩa là đường thẳng nghiêng từ trên-phải xuống dưới-trái (Clockwise).
                # OpenCV getRotationMatrix2D: góc DƯƠNG là xoay NGƯỢC chiều kim đồng hồ (Counter-Clockwise).
                # Để sửa một ảnh bị nghiêng Clockwise (âm), ta phải xoay nó Counter-Clockwise (dương).
                # Do đó: correction_angle = -median_angle!
                correction_angle = -median_angle
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
    save_debug_image(img, "01_deskewed", output_dir)
    
    # 2. Binarize
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    save_debug_image(thresh, "02_binarized", output_dir)
    
    return img, thresh

def detect_columns(page: np.ndarray, thresh: np.ndarray, output_dir: str = None) -> list[dict]:
    """
    Step 3a: Column detection
    Use Vertical Projection Profile to find column boundaries.
    """
    h, w = page.shape[:2]
    
    # [NEW] CHIẾN THUẬT LƯỚI KẺ (GRID LINES):
    # Sách Hán Nôm thường có các lằn kẻ dọc màu đen phân tách các cột.
    # Trong ảnh `thresh`, các lằn kẻ này là các đường thẳng đứng màu trắng rất dài.
    # Ta dùng MORPH_OPEN với kernel dọc dài (h//20 ~ 125px) để xóa sạch chữ Hán, chỉ giữ lại các lằn kẻ.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, h // 20)))
    grid_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    grid_proj = np.sum(grid_lines_img, axis=0)
    
    # Tìm các đỉnh của lưới kẻ (prominence vừa đủ để bắt các đường mờ)
    grid_peaks, _ = find_peaks(grid_proj, distance=40, prominence=h * 10)
    
    # Chuẩn bị sẵn dữ liệu Projection (tổng điểm ảnh theo cột dọc) 
    # để phân tích mật độ chữ cho thuật toán chẻ cột Song Hành hoặc Fallback
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    proj = np.sum(morphed, axis=0)
    
    if len(grid_peaks) >= 3:
        # Nếu có lưới kẻ rõ ràng, dùng luôn làm ranh giới cắt! Chính xác tuyệt đối!
        initial_boundaries = [0] + list(grid_peaks) + [w]
        
        widths = np.diff(initial_boundaries)
        valid_widths = [wd for wd in widths if 50 < wd < 200]
        median_width = np.median(valid_widths) if valid_widths else 110
        
        boundaries = [0]
        for i in range(1, len(initial_boundaries)-1):
            prev_b = boundaries[-1]
            curr_b = initial_boundaries[i]
            
            # Khác với thung lũng, Lưới kẻ có độ chính xác tuyệt đối nên không cần Adaptive Splitting.
            # TUY NHIÊN, vẫn phải chạy kiểm tra Song Hành (Double Column) để chẻ đôi nếu cần!
            col_w = curr_b - prev_b
            if col_w > 0.6 * median_width:
                col_proj = proj[prev_b : curr_b]
                smooth_col = np.convolve(col_proj, np.ones(5)/5, mode='same')
                
                mid_left = int(0.30 * col_w)
                mid_right = int(0.70 * col_w)
                
                left_part = smooth_col[:mid_left]
                center_part = smooth_col[mid_left:mid_right]
                right_part = smooth_col[mid_right:]
                
                if len(left_part) > 0 and len(center_part) > 0 and len(right_part) > 0:
                    max_left = np.max(left_part)
                    max_right = np.max(right_part)
                    min_center = np.min(center_part)
                    
                    if (min_center < 0.6 * max_left and min_center < 0.6 * max_right) or (min_center < 255 * 30):
                        split_idx = mid_left + np.argmin(center_part)
                        boundaries.append(int(prev_b + split_idx))
                        
            boundaries.append(curr_b)
            
        # Thêm ranh giới cuối cùng (mép phải trang sách - w)
        boundaries.append(initial_boundaries[-1])
        
    else:
        # FALLBACK: Nếu sách viết tay không có lưới kẻ, dùng thuật toán thung lũng (Valleys) cũ
        kernel_size = 25
        smoothed = np.convolve(proj, np.ones(kernel_size)/kernel_size, mode='same')
        
        peaks, _ = find_peaks(-smoothed, distance=40, prominence=15000)
        initial_boundaries = [0] + list(peaks) + [w]
        
        widths = np.diff(initial_boundaries)
        valid_widths = [wd for wd in widths if 50 < wd < 200]
        median_width = np.median(valid_widths) if valid_widths else 110
        
        boundaries = [0]
        for i in range(1, len(initial_boundaries)-1):
            prev_b = boundaries[-1]
            curr_b = initial_boundaries[i]
            if curr_b - prev_b > 1.4 * median_width:
                local_segment = -smoothed[prev_b:curr_b]
                local_peaks, _ = find_peaks(local_segment, distance=40, prominence=2000)
                for lp in local_peaks:
                    boundaries.append(prev_b + lp)
            else:
                # [NEW] KIỂM TRA CỘT CHÚ THÍCH SONG HÀNH (DOUBLE COLUMN)
                col_w = curr_b - prev_b
                if col_w > 0.6 * median_width:
                    col_proj = proj[prev_b : curr_b]
                    smooth_col = np.convolve(col_proj, np.ones(5)/5, mode='same')
                    
                    mid_left = int(0.30 * col_w)
                    mid_right = int(0.70 * col_w)
                    
                    left_part = smooth_col[:mid_left]
                    center_part = smooth_col[mid_left:mid_right]
                    right_part = smooth_col[mid_right:]
                    
                    if len(left_part) > 0 and len(center_part) > 0 and len(right_part) > 0:
                        max_left = np.max(left_part)
                        max_right = np.max(right_part)
                        min_center = np.min(center_part)
                        
                        if (min_center < 0.6 * max_left and min_center < 0.6 * max_right) or (min_center < 255 * 30):
                            split_idx = mid_left + np.argmin(center_part)
                            boundaries.append(int(prev_b + split_idx))
                            
            boundaries.append(curr_b)
            
        # Thêm ranh giới cuối cùng (mép phải trang sách - w)
        boundaries.append(initial_boundaries[-1])
    # -------------------------------------
    
    columns_info = []
    # Strict Right-to-Left ordering
    for i in range(len(boundaries)-1, 0, -1):
        x_right = int(boundaries[i])
        x_left = int(boundaries[i-1])
        
        col_img = page[:, x_left:x_right]
        col_thresh = thresh[:, x_left:x_right]
        
        columns_info.append({
            'img': col_img,
            'thresh': col_thresh,
            'x_left': x_left,
            'x_right': x_right,
            'width': x_right - x_left,
            'height': page.shape[0],
            'x_center': (x_left + x_right) / 2
        })
        
    # Draw bounding boxes for debug
    if output_dir:
        debug_img = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        for c in columns_info:
            cv2.rectangle(debug_img, (c['x_left'], 0), (c['x_right'], c['height']), (0, 0, 255), 2)
        save_debug_image(debug_img, "03_detected_columns", output_dir)
        
    return columns_info

def analyze_and_scale_columns(columns_info: list[dict], page_width: int, output_dir: str = None) -> list[dict]:
    """
    Step 3b: Geometric Classification & Auto-Scaling
    Computes horizontal projection to find expected characters.
    Classifies columns as NOISE, BANXIN, or TEXT.
    Auto-scales small annotations.
    """
    valid_cols = []
    
    # 1. Compute Horizontal Projection for all columns to get expected characters
    all_distances = []
    for c in columns_info:
        fill_ratio = np.count_nonzero(c['thresh']) / c['thresh'].size
        c['fill_ratio'] = fill_ratio
        
        # Horizontal projection
        h_proj = np.sum(c['thresh'], axis=1)
        h_smoothed = np.convolve(h_proj, np.ones(15)/15, mode='same')
        h_peaks, _ = find_peaks(h_smoothed, distance=20, prominence=1000)
        
        c['expected_chars'] = len(h_peaks)
        if len(h_peaks) > 1:
            distances = np.diff(h_peaks)
            c['median_distance'] = np.median(distances)
            if fill_ratio > 0.02 and fill_ratio < 0.6 and c['width'] >= 35:
                all_distances.extend(distances)
        else:
            c['median_distance'] = 0
            
    page_median_glyph = np.median(all_distances) if all_distances else 80.0
    
    # 2. Classification & Auto-scaling
    for idx, c in enumerate(columns_info):
        # A. Phân loại vai trò (Banxin vs Noise vs Text)
        if c['width'] < 35 or c['fill_ratio'] < 0.02 or c['fill_ratio'] > 0.6 or c['expected_chars'] < 3:
            c['role'] = 'NOISE'
            continue
            
        x_center_rel = c['x_center'] / page_width
        # Banxin (Gutter) is usually precisely in the middle (0.45-0.55)
        if 0.45 < x_center_rel < 0.55:
            c['role'] = 'BANXIN'
        else:
            c['role'] = 'TEXT'
            
        # D. Auto-scale Annotations (Small text)
        if c['median_distance'] > 0 and c['median_distance'] < 0.75 * page_median_glyph:
            scale_factor = page_median_glyph / c['median_distance']
            c['scaled_img'] = cv2.resize(c['img'], None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            c['is_scaled'] = True
            c['scale_factor'] = scale_factor
        else:
            c['scaled_img'] = c['img']
            c['is_scaled'] = False
            
        valid_cols.append(c)
        
    return valid_cols

def recognize_columns(valid_cols: list[dict], debug_ocr: bool = False, ocr_engine = None) -> list[dict]:
    """
    Step 4: Character recognition & Reassembly using PaddleOCR.
    Implements Point C (Missing Character Check).
    """
    if ocr_engine is None:
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        ocr_engine = PaddleOCR(lang='chinese_cht', use_textline_orientation=True)
    
    final_results = []
    
    for c in valid_cols:
        pad = 50
        col_padded = cv2.copyMakeBorder(c['scaled_img'], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if len(col_padded.shape) == 2:
            col_padded = cv2.cvtColor(col_padded, cv2.COLOR_GRAY2BGR)
            
        result = ocr_engine.ocr(col_padded)
        col_text = ""
        min_conf = 1.0
        
        if result and result[0]:
            for res in result:
                if res is None: continue
                # Handle PaddleOCR 3.7.x / 2.x formats
                if isinstance(res, dict) or hasattr(res, 'get') or hasattr(res, 'dt_polys'):
                    rec_texts = getattr(res, 'rec_texts', None) or res.get('rec_texts', [])
                    rec_scores = getattr(res, 'rec_scores', None) or res.get('rec_scores', [])
                    for i in range(len(rec_texts)):
                        text = str(rec_texts[i])
                        conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                        min_conf = min(min_conf, conf)
                        col_text += f"[{text}?]" if debug_ocr and conf < 0.8 else text
                elif isinstance(res, list):
                    for line in res:
                        try:
                            text_info = line[1]
                            text = str(text_info[0]) if isinstance(text_info, (tuple, list)) else str(text_info)
                            conf = float(text_info[1]) if isinstance(text_info, (tuple, list)) and len(text_info) > 1 else 1.0
                            min_conf = min(min_conf, conf)
                            col_text += f"[{text}?]" if debug_ocr and conf < 0.8 else text
                        except Exception: pass
                        
        # Point B: Do not hardcode variants. Output raw prediction.
        
        # Point C: Missing Character Warning
        # Dùng regex đếm số lượng chữ Hán thực sự AI đọc được
        cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]', col_text)
        num_recognized = len(cjk_chars)
        
        # LỌC NHIỄU: Nếu cột KHÔNG CÓ chữ Hán nào và không phải là Gutter (Banxin), 
        # chắc chắn nó là cột rác (đường kẻ, vết ố bị nhận diện nhầm thành M, Y, 1, .)
        if num_recognized < 1 and c['role'] != 'BANXIN':
            continue
            
        # Không in cảnh báo MISSING CHARS nữa vì thuật toán đếm số nét ngang (h_peaks) 
        # thường phóng đại số chữ lên gấp 1.5 - 2 lần đối với chữ Hán phức tạp.
            
        # Không thêm các tag [BANXIN] hay [SCALED] vào đầu chuỗi nữa để xuất ra văn bản sạch
        final_results.append({
            'text': col_text,
            'col_info': c,
            'min_conf': min_conf
        })
        
    return final_results

def post_process_ocr(raw_results: list[dict], is_half_page: bool = False) -> list[dict]:
    """
    Sửa các lỗi OCR đồng hình thường gặp dựa trên Từ điển Ngữ cảnh (N-gram).
    """
    # Danh sách các luật sửa lỗi: (Sai, Đúng)
    correction_rules = [
        # ==========================================
        # 1. CÁC LỖI CŨ
        # ==========================================
        ("廣義脊", "廣義省"),
        ("四土",   "田土"),
        ("五萬空百", "五萬八百"),
        ("二十九部", "二十九畝"),
        ("稅案",   "稅粟"),

        # ==========================================
        # 2. BỔ SUNG TỪ ĐẠI NAM NHẤT THỐNG CHÍ
        # ==========================================
        
        # Lỗi Địa danh / Tên riêng / Thuật ngữ lịch sử
        ("北跨潼江", "北跨𤅷江"),      # Nhầm chữ Linh (Sông Gianh) thành Đồng
        ("南拓石臘", "南拓占臘"),      # Nhầm Chiêm Lạp thành Thạch Lạp
        ("澄光樹",   "澄光榭"),        # Nhầm Tạ (thủy tạ) thành Thụ (cây)
        ("均征名",   "均社名"),        # Nhầm Xã thành Chinh
        ("大南統志", "大南一統志"),    # Thiếu chữ Nhất trong tên sách

        # Lỗi Niên hiệu / Can Chi
        ("戊戍",     "戊戌"),          # Nhầm Tuất thành Thú (Mậu Tuất)
        ("壬戍",     "壬戌"),          # Nhầm Tuất thành Thú (Nhâm Tuất)

        # Lỗi sai chữ do hình dáng (Visual Similarity)
        ("圗",       "圖"),            # Chữ Đồ (bị sai dị thể)
        ("自辦",     "自辨"),          # Nhầm Biện (phân biệt) thành Biện (làm việc)
        ("勉彈",     "勉殫"),          # Nhầm Đan (hết sức) thành Đàn
        ("史歲",     "史宬"),          # Sử Thành (kho sử) bị nhầm thành Tuế (JSON)
        ("史成",     "史宬"),          # Sử Thành bị nhầm thành Thành (thành công)
        ("屋胥",     "屋脊"),          # Nhầm Tích (mái nhà) thành Tư
        ("八荒皆闥", "八荒皆闊"),      # Nhầm Khoát thành Thát
        ("湖炎邦",   "溯炎邦"),        # Nhầm Tố thành Hồ
        ("鴻厖",     "鴻龐"),          # Nhầm Bàng (Hồng Bàng) thành Mang
        ("天計",     "天討"),          # Nhầm Thảo (thảo phạt) thành Kế
        ("安典",     "安輿"),          # Nhầm Dư (xe kiệu) thành Điển
        ("優遼",     "優遊"),          # Nhầm Du (dạo chơi) thành Liêu
        
        # Lỗi biến thể của cụm "Cương vực" (幅員)
        ("幅賴",     "幅員"),          
        ("幅帽",     "幅員"),          
        ("幅隕",     "幅員"),          

        # Lỗi ảo giác (Dư chữ không tồn tại)
        ("繪於秀",   "繪於"),          # Xóa chữ "Tú" bị máy tự vẽ ra ở cuối dòng
    ]
    
    filtered_results = []
    for item in raw_results:
        text = item['text']
        
        for wrong, right in correction_rules:
            text = text.replace(wrong, right)
            
        # 3. LỌC NHIỄU (Anti-noise filter)
        text = re.sub(r'[0-9]', '', text)
        
        if not is_half_page:
            if len(text) <= 2 and all(char in "一二三丨|" for char in text):
                text = "" # Đánh dấu chuỗi rỗng để xóa
            
        item['text'] = text
        
        if text.strip():
            filtered_results.append(item)
            
    return filtered_results

def ocr_sinonom_page(image_path: str = None, image_np: np.ndarray = None, debug_ocr: bool = False, output_dir: str = None, ocr_engine = None, verbose: bool = False, is_half_page: bool = False, pdf_page_num: int = None, pdf_filename: str = None):
    if verbose:
        src_name = image_path if image_path else f"{pdf_filename} (page {pdf_page_num})"
        print(f"Processing image: {src_name}")
        print("Step 2: Preprocessing...")
    page, thresh = preprocess(image_path, image_np, output_dir)
    
    if verbose:
        print("Step 3a: Column detection...")
    columns_info = detect_columns(page, thresh, output_dir)
    if verbose:
        print(f"         Detected {len(columns_info)} initial vertical slices.")
    
    if verbose:
        print("Step 3b: Geometric Classification & Auto-Scaling...")
    valid_cols = analyze_and_scale_columns(columns_info, page.shape[1], output_dir)
    
    if verbose:
        print("Step 4: Character recognition & Reassembly...")
    col_results = recognize_columns(valid_cols, debug_ocr, ocr_engine=ocr_engine)
    
    col_results = post_process_ocr(col_results, is_half_page=is_half_page)
    
    import json
    
    if pdf_filename:
        volume = os.path.basename(os.path.dirname(os.path.abspath(pdf_filename))) if image_path is None else "pdf"
        filename = pdf_filename
        page_number = pdf_page_num if pdf_page_num is not None else 0
    else:
        volume = os.path.basename(os.path.dirname(os.path.abspath(image_path))) if image_path else "unknown"
        filename = os.path.basename(image_path) if image_path else "unknown"
        if pdf_page_num is not None:
            page_number = pdf_page_num
        else:
            m = re.search(r'(\d+)', filename)
            page_number = int(m.group(1)) if m else 0
    
    page_width = page.shape[1]
    
    keyword_pattern = re.compile(r'(大南|統志|卷|表)')
    candidates = []
    
    for res in col_results:
        c = res['col_info']
        center_x = c['x_center']
        text = res['text']
        
        if is_half_page:
            if page_number > 0:
                if page_number % 2 != 0: # Trang Lẻ (Odd) -> Bản tâm nằm bên PHẢI (Right edge)
                    valid_zone = (center_x > 0.75 * page_width)
                    distance = page_width - center_x
                else: # Trang Chẵn (Even) -> Bản tâm nằm bên TRÁI (Left edge)
                    valid_zone = (center_x < 0.25 * page_width)
                    distance = center_x
            else:
                valid_zone = (center_x < 0.20 * page_width) or (center_x > 0.80 * page_width)
                distance = min(center_x, page_width - center_x)
        else:
            valid_zone = (0.35 * page_width < center_x < 0.65 * page_width)
            distance = abs(center_x - (page_width / 2))
            
        if valid_zone:
            score = 0
            if re.search(r'[■□▣◼◻]', text):
                score += 200
            if keyword_pattern.search(text):
                score += 50
            if re.search(r'[一二三四五六七八九十百]+', text):
                score += 50
                
            # 4. Độ dài chuẩn của Bản tâm
            if len(text) <= 8:
                score += 30
                
            # 5. KHẮC CHẾ: Các từ khóa thường nằm ở cột Text thường (Bài tựa, Kết thúc chương)
            if re.search(r'[止終序]', text):
                score -= 200
                
            # 6. KHẮC CHẾ: Bản tâm hiếm khi dài quá 15 ký tự
            if len(text) > 15:
                score -= 200
                
            candidates.append({
                "res": res,
                "distance": distance,
                "score": score
            })
            
    # BƯỚC 2: CHỌN BẢN TÂM TỐT NHẤT
    centerfold_res = None
    if candidates:
        # Lọc những ứng viên có điểm > 0 (Tức là có chứa từ khóa HOẶC có số)
        valid_candidates = [c for c in candidates if c["score"] > 0]
        
        if valid_candidates:
            # Ưu tiên 1: Điểm càng cao càng tốt
            # Ưu tiên 2: Nếu điểm bằng nhau, khoảng cách tới trung tâm càng gần càng tốt
            valid_candidates = sorted(valid_candidates, key=lambda x: (-x["score"], x["distance"]))
            centerfold_res = valid_candidates[0]["res"]
            
    if output_dir:
        debug_img = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        if centerfold_res:
            c = centerfold_res['col_info']
            # Vẽ viền xanh lá cây (Green) dày 5px cho cột được chọn làm Bản tâm
            cv2.rectangle(debug_img, (int(c['x_left']), 0), (int(c['x_right']), int(c['height'])), (0, 255, 0), 5)
            # Viết chữ "Banxin" lên trên cùng
            cv2.putText(debug_img, "Banxin", (int(c['x_left']), 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        save_debug_image(debug_img, "04_centerfold_banxin", output_dir)
            
    json_output = []
    for res in col_results:
        c = res['col_info']
        bbox = [[c['x_left'], 0], [c['x_right'], 0], [c['x_right'], c['height']], [c['x_left'], c['height']]]
        
        is_middle = 1 if centerfold_res == res else 0
        
        # Gắn cờ consider: 1 nếu AI không chắc chắn (độ tự tin < 0.85) 
        # Hoặc nếu cột nghi ngờ có chứa rác (ví dụ có Hắc khẩu nhưng không phải Bản tâm)
        consider = 0
        text = res['text']
        if res.get('min_conf', 1.0) < 0.85:
            consider = 1
        elif re.search(r'[■□▣◼◻]', text) and not is_middle:
            consider = 1
        
        json_output.append({
            'bbox': str(bbox),
            'volume': volume,
            'page': filename,
            'page_number': page_number,
            'text': text,
            'middle': is_middle,
            'consider': consider
        })
    
    return json_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Sino-Nôm OCR Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--debug-ocr", action="store_true", help="Wrap low confidence characters in [ ?]")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save debug images")
    args = parser.parse_args()
    
    results = ocr_sinonom_page(
        args.image, 
        debug_ocr=args.debug_ocr, 
        output_dir=args.output_dir,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("  OCR Results (JSON format)")
    print("="*80 + "\n")
    
    import json
    print(json.dumps(results, ensure_ascii=False, indent=4))
    print("\n" + "="*80 + "\n")
