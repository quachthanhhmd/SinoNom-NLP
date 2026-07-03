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

def preprocess(image_path: str, output_dir: str = None, max_size: int = 2500):
    """
    Step 2: Preprocessing
    Deskew using Hough line detection against vertical rules.
    Binarize with Adaptive Thresholding.
    Nếu ảnh quá to (vượt quá max_size), tự động thu nhỏ lại để tránh nhiễu nét và tăng tốc OCR.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    h, w = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # 1. Deskew using Standard Hough Transform (more accurate sub-degree angles than HoughLinesP)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # Use high resolution for theta (0.1 degrees = np.pi/1800)
    lines = cv2.HoughLines(edges, 1, np.pi / 1800, 200)
    
    if lines is not None:
        vertical_angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            # Find vertical lines (angle near 0 or 180 degrees in Hough space)
            if angle < 45:
                vertical_angles.append(angle)
            elif angle > 135:
                vertical_angles.append(angle - 180)
                
        if vertical_angles:
            median_angle = np.median(vertical_angles)
            if abs(median_angle) > 0.1:  # Rotate if skew is > 0.1 degrees
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    proj = np.sum(morphed, axis=0)
    kernel_size = 25
    smoothed = np.convolve(proj, np.ones(kernel_size)/kernel_size, mode='same')
    
    # Sử dụng prominence lớn hơn (10000 thay vì 3000) để bỏ qua các thung lũng nông 
    # Sử dụng prominence = 15000 để lờ đi các nhiễu rãnh dọc BÊN TRONG lòng chữ (do các bộ thủ trái-phải tạo ra),
    # đồng thời vẫn giữ được các khe hở thật sự giữa các cột chữ (thường > 25000).
    peaks, _ = find_peaks(-smoothed, distance=40, prominence=15000)
    initial_boundaries = [0] + list(peaks) + [page.shape[1]]
    
    # --- THUẬT TOÁN ADAPTIVE SPLITTING ---
    # Phục hồi các cột bị bỏ sót (VD: cột chỉ có 1 chữ "論") do quá hẹp và ít mực
    widths = np.diff(initial_boundaries)
    valid_widths = [w for w in widths if 50 < w < 200]
    median_width = np.median(valid_widths) if valid_widths else 110
    
    boundaries = [0]
    for i in range(1, len(initial_boundaries)-1):
        prev_b = boundaries[-1]
        curr_b = initial_boundaries[i]
        
        # Nếu phát hiện khoảng cách giữa 2 lằn cắt quá rộng (> 1.4 lần cột bình thường)
        if curr_b - prev_b > 1.4 * median_width:
            # Quét lại cục bộ với độ nhạy (prominence) cực thấp để mò ra rãnh cắt ẩn
            local_segment = -smoothed[prev_b:curr_b]
            local_peaks, _ = find_peaks(local_segment, distance=40, prominence=2000)
            for lp in local_peaks:
                boundaries.append(prev_b + lp)
                
        boundaries.append(curr_b)
        
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

def post_process_ocr(raw_results: list[dict]) -> list[dict]:
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
        # Xóa toàn bộ số từ 0-9 lẫn vào trong chuỗi Hán văn do máy nhận diện nhầm
        text = re.sub(r'[0-9]', '', text)
        
        # Lọc rác đường viền (Border Hallucinations)
        # Máy hay nhận diện đường thẳng kẻ khung thành chữ "一", "二", "丨"
        # Nếu cột chỉ có 1-2 ký tự và toàn là các nét đơn giản này -> Khẳng định là rác
        if len(text) <= 2 and all(char in "一二三丨|" for char in text):
            text = "" # Đánh dấu chuỗi rỗng để xóa
            
        # Cập nhật lại text sau khi dọn dẹp
        item['text'] = text
        
        # Chỉ giữ lại những cột có chữ sau khi dọn rác
        if text.strip():
            filtered_results.append(item)
            
    return filtered_results

def ocr_sinonom_page(image_path: str, debug_ocr: bool = False, output_dir: str = None, ocr_engine = None, verbose: bool = False):
    if verbose:
        print(f"Processing image: {image_path}")
        print("Step 2: Preprocessing...")
    page, thresh = preprocess(image_path, output_dir)
    
    if verbose:
        print("Step 3a: Column detection...")
    columns_info = detect_columns(page, thresh, output_dir)
    if verbose:
        print(f"         Detected {len(columns_info)} initial vertical slices.")
    
    if verbose:
        print("Step 3b: Geometric Classification & Auto-Scaling...")
    valid_cols = analyze_and_scale_columns(columns_info, page.shape[1], output_dir)
    if verbose:
        print(f"         Kept {len(valid_cols)} valid text columns (Filtered out {len(columns_info) - len(valid_cols)} noise columns).")
    
    if verbose:
        print("Step 4: Character recognition & Reassembly...")
    col_results = recognize_columns(valid_cols, debug_ocr, ocr_engine=ocr_engine)
    
    # --- Áp dụng Từ điển Ngữ cảnh (N-gram) để sửa các lỗi nhận diện ---
    col_results = post_process_ocr(col_results)
    
    # Generate JSON structures
    import json
    volume = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
    filename = os.path.basename(image_path)
    m = re.search(r'(\d+)', filename)
    page_number = int(m.group(1)) if m else 0
    
    page_width = page.shape[1]
    
    keyword_pattern = re.compile(r'(大南|統志|卷|表)')
    candidates = []
    
    # BƯỚC 1: LỌC ỨNG VIÊN BẢN TĂM
    # Bản tâm không nhất thiết phải nằm gắt gao ở 45-55%, có thể trang bị crop lệch
    # Nên mở rộng ra 35% - 65%
    for res in col_results:
        c = res['col_info']
        center_x = c['x_center']
        text = res['text']
        
        if 0.35 * page_width < center_x < 0.65 * page_width:
            distance = abs(center_x - (page_width / 2))
            
            # Tính điểm khả năng là Bản tâm
            score = 0
            
            # 1. Ngư vĩ (Fishtail) là dấu hiệu mạnh nhất
            if re.search(r'[■□▣◼◻]', text):
                score += 200
                
            # 2. Từ khóa tên sách
            if keyword_pattern.search(text):
                score += 50
                
            # 3. Số trang
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
