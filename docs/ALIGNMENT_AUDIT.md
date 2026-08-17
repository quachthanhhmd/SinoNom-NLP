# Audit pipeline tạo parallel corpus Hán–Việt và kế hoạch khắc phục

- Ngày audit: 2026-08-17
- Repository: `SinoNom-NLP`
- Branch: `features/mapping-translation`
- Baseline: `7b52a71cf1eef80640191d2866fb2a45d3553f22`
- Phạm vi ban đầu của audit: tài liệu; trạng thái hiện tại đã có remediation implementation đi kèm
- Artifact đối chiếu: `output/HVB_001/HVB_001_parallel.xlsx`, TSV tương ứng và các cache Phase 1/2/3

## Trạng thái khắc phục sau audit

Các finding bên dưới mô tả baseline đã tạo artifact cũ. Production code hiện đã được sửa ở các điểm đã chứng minh được bằng test:

- thay mock mặc định bằng ensemble + true monotonic m–n decoder (`m,n ≤ 3`), có merge/skip/length prior và hard volume boundary;
- giữ một CSV row là một source unit, không hard-split theo mọi whitespace; newline đơn là soft layout boundary;
- nhận diện q8/q9 theo nội dung/manifest và tách 10→11, 16→17 theo heading thật, không theo prefix ID;
- cache theo fingerprint input/config/schema và Phase 3 final không bị xử lý lại;
- Phase 3 chỉ chấp nhận reference hợp lệ, đủ coverage, duy nhất, liên tiếp và đơn điệu; cụm quá lớn/invalid được giữ để review;
- export tách `accepted`, `review`, `unmatched`, giữ provenance; downstream chỉ đọc accepted và deduplicate;
- regression suite hiện có 13 test cho decoder, invariant, boundary, ordering, segmentation và LLM parser.

Việc sửa code **không biến artifact `HVB_001_parallel.xlsx` cũ thành đúng**. Cần chạy lại pipeline để tạo corpus phiên bản mới, sau đó đánh giá trên gold sample trước khi khẳng định tỷ lệ đúng thực tế.

## 1. Executive summary

Pipeline hiện tại **chưa đủ an toàn để xuất bản hoặc dùng trực tiếp làm dữ liệu huấn luyện**. Lỗi không đến từ một điểm duy nhất mà là chuỗi lỗi dữ liệu và kiến trúc:

1. **Metadata quyển sai nhưng nội dung không mất như suy luận ban đầu.** Kiểm tra tiêu đề nội dung xác nhận `q8_sentences.csv` là quyển 8 dù ID mang prefix `Q9_*`; `q9_sentences.csv` là quyển 9 dù ID mang prefix `Q8_*`. `q10_11_sentences.csv` có cả tiêu đề quyển 10 và 11; `q16_17_sentences.csv` có cả quyển 16 và 17. Vì vậy không được rename/swap file mù hoặc kết luận Q11/Q17 bị thiếu theo ID. Lỗi thật là code tin prefix ID để chia quyển, khiến toàn bộ phần quyển 11/17 bị gán sai volume.
2. **Artifact cuối chứa nguyên output Phase 3, kể cả blank.** `HVB_001_parallel.tsv` có 15.282 dòng và khớp chính xác 15.282/15.282 dòng khi nối các cache `*_phase3.json` theo thứ tự group. Trong đó chỉ 10.409 dòng có đủ Hán và Việt (68,1%); 3.188 dòng chỉ có Việt và 1.685 dòng chỉ có Hán.
3. **Phase 3 có thể phá tính đơn điệu và tính duy nhất của index.** Parser tin output LLM nhưng không kiểm tra coverage, uniqueness hoặc order. Cache thực tế có duplicate/backtrack/gap của `han_indices`; có cặp merge đến 74 câu Hán mặc dù Phase 1 đặt trần 15.
4. **Cache không gắn fingerprint đầu vào/cấu hình/model và resume không idempotent.** Cache Phase 3 được ưu tiên sử dụng dù cờ `--realign` có bật hay không; khi bật `--realign`, cache đã realign vẫn có thể bị quét và biến đổi lại.
5. **Có hai pipeline khác nhau nhưng tài liệu/vận hành chưa phân biệt rõ.** `main.py` vẫn dùng `BERTAlignerWrapper`, thực chất chỉ ghép theo index và blank-padding; `run_mapping.py` dùng ensemble + DP + LLM. Chạy nhầm entry point sẽ tạo corpus sai theo hai cơ chế khác nhau.
6. **Segmentation và DP đang khuếch đại lỗi boundary.** Phía Hán bị hard-split theo newline/whitespace; phía Việt lại xóa newline. DP chỉ hỗ trợ `k↔1` và `1↔l`, không phải monotonic `m↔n` tổng quát, không có anchor, band, length/merge prior hay resynchronization.

Khuyến nghị tuần tới là **không tinh chỉnh threshold trên pipeline hiện tại trước**. Cần sửa theo thứ tự: khóa manifest/mapping và ordering → giữ provenance + soft segmentation → monotonic span alignment có anchor/resync → confidence/review/export gate → rerun corpus từ cache sạch có fingerprint.

## 2. Phạm vi audit

Đã kiểm tra:

- Entry points và orchestration: `main.py`, `core/pipeline.py`, `run_mapping.py`.
- Segmentation và load dữ liệu: `nlp/segmenter.py`, `run_mapping.py::load_sino_csv`, các CSV trong `dataset/MAPPING`.
- Alignment: mock aligner, embedding aligner, ensemble scorers, fusion và DP trong `nlp/aligner.py` và `nlp/scorers/*`.
- Phase 2/3: `nlp/qwen_verifier.py`, `nlp/qwen_realigner.py`.
- OCR và ordering: `ocr/providers.py`, `ocr/ensemble.py`, thứ tự file/page/row ở orchestration.
- Export và downstream dataset: `utils/exporters.py`, `scripts/prepare_data.py`, notebook Kaggle.
- Artifact thực tế: workbook/TSV HVB_001 và toàn bộ cache Phase 1/2/3.

Không làm trong audit này:

- Không chạy lại model embedding/Qwen/Gemini.
- Không sửa production logic, dữ liệu nguồn hoặc workbook.
- Không kết luận chất lượng dịch bằng cảm quan cho toàn bộ 10.409 cặp; các kết luận semantic cần gold sample ở Phase 0.
- Không xác minh được provenance OCR trước khi các CSV `dataset/MAPPING` được tạo, vì repo không chứa pipeline tái tạo đầy đủ các CSV đó từ ảnh/PDF.

## 3. Sơ đồ pipeline thực tế

Repo có hai đường chạy độc lập:

```text
main.py
  ảnh Hán -> OCR -> hard segmentation Hán
  PDF Việt -> PyPDF2 -> segmentation Việt
  -> BERTAlignerWrapper (mock index 1-1 + blank padding)
  -> TSV/XLSX

run_mapping.py
  Sino CSV + Viet CSV
  -> lọc/clean + hard split Hán theo whitespace
  -> Phase 1 ensemble + global DP
  -> Phase 2 LLM verify, reject thành blank rows
  -> Phase 3 LLM local realign blank clusters
  -> chia lại volume theo han_indices/current_vol
  -> hierarchical exporter
  -> artifact merged HVB_001 được tạo ngoài đường export đã commit
```

`HVB_001_parallel.xlsx` thuộc đường `run_mapping.py`/Phase 3, không phải output của mock aligner trong `main.py`. Tuy nhiên mock aligner vẫn là lỗi blocker vì README coi `main.py` là đường chạy chính.

## 4. Bằng chứng từ artifact HVB_001

### 4.1. Thống kê tổng thể

Workbook có một sheet, vùng dữ liệu `A1:C15283` (1 header + 15.282 dòng):

| Loại dòng | Số lượng | Tỷ lệ |
|---|---:|---:|
| Có cả Hán và Việt | 10.409 | 68,1% |
| Chỉ Việt | 3.188 | 20,9% |
| Chỉ Hán | 1.685 | 11,0% |
| Tổng | 15.282 | 100% |

Đối chiếu theo thứ tự group trong `run_mapping.py`, phép nối 11 cache `*_phase3.json` tạo đúng 15.282 dòng và **khớp nội dung Hán/Việt 15.282/15.282 dòng** với TSV/workbook cuối. Điều này xác nhận artifact cuối là bản full Phase 3 có blank, không phải bản clean đã lọc của `export_hierarchical()`.

### 4.2. Pattern lỗi quan sát được

- Ngay đầu corpus, dòng 5 chỉ có Việt. Dòng trước đó đã ghép một câu Hán dài với câu Việt chứa phần nội dung chỉ tương ứng một phần; đây là pattern boundary `1↔2` bị tách thành `1↔1` + `blank↔1`.
- Vùng 499–511 tiếp tục có các chuỗi `Hán↔Việt`, `blank↔Việt`, `Hán↔blank`, phù hợp với boundary mismatch và skip xen kẽ.
- Vùng 999–1.011 có thơ/chú giải bị ghép lệch: nhiều câu Việt liên tiếp đứng blank trước khi Hán xuất hiện, sau đó chú giải và câu thơ bị kéo lệch dây chuyền.
- Vùng khoảng 7.999–8.011 có câu Việt rất dài ghép với fragment Hán ngắn, rồi xen nhiều `Hán↔blank`/`blank↔Việt`; đây là dấu hiệu block-level mismatch, không phải chỉ một câu khó.
- Vùng khoảng 11.999–12.011 có nhiều dòng Hán OCR dài không có Việt xen với các câu Việt mô tả cầu; semantic/boundary không ổn định.

Các chuỗi blank dài nhất đo được gồm:

- 37 dòng chỉ Việt tại 2.594–2.630.
- 33 dòng chỉ Việt tại 5.704–5.736.
- 29 dòng chỉ Việt tại 1.880–1.908 và 1.926–1.954.
- Ba block 29 dòng chỉ Hán quanh 7.611–7.709.
- 26 dòng chỉ Hán tại 10.265–10.290.

Có nhiều outlier độ dài không hợp lý, ví dụ một ký tự Hán ghép với 554 ký tự Việt, hoặc nhiều cặp một–hai ký tự Hán ghép với 197 ký tự Việt. Đây phải là review blocker, nhưng schema cuối không còn confidence/provenance để chặn.

### 4.3. Phase 3 vi phạm index invariant

Phase 1 cache giữ `han_indices` đơn điệu và không lặp. Sau Phase 3, một số group có anomaly:

| Cache Phase 3 | Duplicate Hán index | Backtrack/overlap giữa item | Gap index | Max số Hán trong một pair |
|---|---:|---:|---:|---:|
| `q05_phase3.json` | 3 | 9 | 97 | 38 |
| `q07_08_phase3.json` | 42 | 10 | 0 | 47 |
| `q10_11_phase3.json` | 2 | 6 | 2 | 15 |
| `q14_15_phase3.json` | 11 | 3 | 22 | 46 |
| `q16_17_phase3.json` | 2 | 10 | 0 | 74 |
| `q2_3_4_phase3.json` | 66 | 17 | 0 | 47 |

`max_merge_han=15` chỉ là constraint của Phase 1. Phase 3 không enforce constraint tương đương, nên artifact cuối có thể vượt trần và reuse index.

## 5. Findings theo severity

### S0 — Blocker: chia volume theo metadata ID sai thay vì tiêu đề nội dung

**Bằng chứng code**

- `run_mapping.py:413-460` hard-code mapping theo filename.
- `run_mapping.py:476-538` chỉ tách merged file theo prefix trong cột ID, nhưng không kiểm tra tập prefix kỳ vọng hoặc coverage.

**Bằng chứng dữ liệu**

- `q8_sentences.csv` có tiêu đề `大南一統志卷之八`; đây là quyển 8, chỉ prefix ID `Q9_*` sai.
- `q9_sentences.csv` có tiêu đề `大南一統志卷之九`; đây là quyển 9, chỉ prefix ID `Q8_*` sai.
- `q10_11_sentences.csv` có marker nội dung quyển 11 dù ID tiếp tục là `Q10_*`.
- `q16_17_sentences.csv` có marker nội dung quyển 17 dù ID tiếp tục là `Q16_*`.

**Tác động**

- Group 7–8 và group 9 dùng đúng nội dung quyển theo filename, nhưng provenance ID bị sai.
- Logic tách group 10–11 và 16–17 theo prefix không bao giờ chuyển sang 11/17, nên attribution/export volume sai và hard boundary thật bị mất.
- Lệch block trong artifact vẫn có thật, nhưng không được quy nguyên nhân cho “mất Q11/Q17”; nguyên nhân đã chứng minh là decoder/Phase 3/segmentation và volume attribution không an toàn.

**Khắc phục bắt buộc**

- Dùng filename/manifest để khai báo volume và tiêu đề nội dung để xác minh; giữ prefix ID gốc chỉ như provenance.
- Với file ghép, tách tại marker tiêu đề `卷十一`/`卷十七`; fail fast nếu không tìm thấy đủ hai block.
- Cấm transition alignment vượt qua boundary đã xác minh.

### S0 — Blocker: Phase 3 không bảo toàn coverage, uniqueness và monotonicity

**Bằng chứng code**

- `nlp/qwen_realigner.py:192-251` gom mọi chuỗi blank thành cluster và tách riêng danh sách Hán/Việt.
- `nlp/qwen_realigner.py:322-425` parse reference LLM nhưng không kiểm tra mỗi H/V được dùng đúng một lần, không kiểm tra order, coverage hoặc duplicate.
- `_resolve_text()` ở `nlp/qwen_realigner.py:427-454` còn chấp nhận plain text do LLM tự viết, trái với prompt “chỉ dùng ID”.
- `nlp/qwen_realigner.py:516-546` cắt cluster theo tỷ lệ bằng phép chia nguyên; chunk cuối có thể vượt limit và boundary cắt không dựa trên anchor/semantic.
- `nlp/qwen_realigner.py:633-646` splice output trực tiếp vào path và ghi cache, không chạy validator sau splice.

**Tác động**

- Duplicate/backtrack/gap index đã xuất hiện trong cache thực tế.
- Phase 3 có thể tăng match rate bề ngoài bằng cách reuse hoặc bỏ sót source unit.
- Pair score của output realign được hard-code `0.75` nếu đủ hai phía (`nlp/qwen_realigner.py:416-423`), không phản ánh độ tin cậy thật.

**Khắc phục bắt buộc**

- LLM chỉ đề xuất transition trên tập ID; validator deterministic phải enforce typed schema, coverage, uniqueness, monotonicity, span limit và anchor order.
- Nếu proposal invalid: reject toàn cluster, giữ trạng thái review; tuyệt đối không splice partial output.
- Phase 3 phải là hàm idempotent trên cùng input + config.

### S0 — Blocker: cache không có provenance/fingerprint và có thể tái xử lý Phase 3

**Bằng chứng code**

- `run_mapping.py:205-245` đặt tên cache chỉ theo basename của file Việt và phase; không có hash của Hán, Việt, cleaner, config, model hoặc code revision.
- `run_mapping.py:219-225` luôn nạp Phase 3 nếu file tồn tại, bất kể cờ vận hành.
- `run_mapping.py:293-305` vẫn gọi realigner khi `--realign` bật.
- `nlp/qwen_realigner.py:476-488` realigner lại nạp chính Phase 3 cache rồi tiếp tục detect cluster.
- `nlp/qwen_verifier.py:248-270` map resume theo tuple text `(han_sentence, viet_sentence)`, nên câu trùng nội dung có thể dùng chung verification result.

**Tác động**

- Đổi input/mapping/threshold/model nhưng vẫn có thể tái dùng cache cũ.
- Chạy lại có thể biến đổi output Phase 3 thêm lần nữa.
- Không thể tái lập chắc chắn artifact cuối từ baseline và tham số đã ghi.

**Khắc phục bắt buộc**

- Cache key phải gồm hash input manifest + normalized units + config + model revision + code schema version.
- Cache immutable theo phase; checkpoint và final cache là hai loại khác nhau.
- Mỗi record dùng stable source ID/span, không map bằng text.

### S0 — Blocker: entry point mặc định dùng aligner mock 1–1

**Bằng chứng code**

- `main.py:62-64` khởi tạo `BERTAlignerWrapper()` và ghi comment “Default to BERT”.
- `nlp/aligner.py:6-23` in rõ `Mock Aligning`, ghép `han[i]` với `viet[i]` và blank-pad đến `max_len`.
- README hướng dẫn người dùng chạy `main.py` và mô tả BERT/m–n, không khớp implementation.

**Tác động**

- Một boundary `1↔2` hoặc `2↔1` làm lệch toàn bộ phần sau.
- Tên class/log/docs tạo cảm giác an toàn giả.

**Khắc phục bắt buộc**

- Vô hiệu hóa đường production này hoặc rename thành `IndexZipBaseline` và yêu cầu flag explicit chỉ dành cho test.
- Một entry point duy nhất phải được coi là canonical.

### S1 — High: segmentation hai phía dùng quy tắc bất đối xứng và phá provenance

**Bằng chứng code**

- `nlp/segmenter.py:10-15`: newline là hard boundary cho Hán.
- `nlp/segmenter.py:44-55`: phía Việt xóa newline trước khi tokenization.
- `core/pipeline.py:59-100`: nối page Hán bằng newline rồi hard-segment, ghi lại thành file mỗi câu một dòng và bỏ blank; page/source boundary bị mất.
- `run_mapping.py:145-164`: Hán CSV bị split tiếp bằng mọi whitespace (`re.split(r'\s+')`), không dựa trên dấu câu hoặc source ID.
- Thực tế một số file bị tăng fragment mạnh: q13 từ 1.132 row lên 1.630 phần; q15 từ 773 lên 1.146; q16_17 từ 2.454 lên 3.079.

**Tác động**

- Hán over-segment trong khi Việt under-segment.
- DP bị ép dùng merge lớn để sửa lỗi do preprocessor tạo ra.
- Không thể trace một pair về page/row/bbox gốc.

**Khắc phục**

- Dùng soft boundary lattice: punctuation/newline/whitespace/page transition là feature có trọng số, không phải hard split duy nhất.
- Giữ stable unit ID và candidate span; không serialize trung gian thành text line làm mất metadata.

### S1 — High: DP không phải true monotonic m–n và thiếu cơ chế resync

**Bằng chứng code**

- `nlp/aligner.py:557-647` chỉ precompute `k↔1` và `1↔l`.
- `nlp/aligner.py:657-724` DP chỉ có skip Hán, skip Việt, `k↔1`, `1↔l`; không có `2↔2`, `2↔3`, v.v.
- `config.py:48-52`: threshold 0,32, skip penalty 0,05, merge Hán đến 15 và Việt đến 2.
- Score match là `similarity - threshold`, không có explicit length prior/merge penalty; merge 1 hay 15 Hán không bị phạt theo độ phức tạp.
- DP chạy global trên cả group, không anchor/band/checkpoint/resynchronization.

**Tác động**

- Merge rất lớn có thể thắng nhiều skip rẻ chỉ nhờ một similarity nhiễu.
- Khi một block sai hoặc thiếu nguyên quyển, path vẫn cố tối ưu toàn cục thay vì fail/resync.
- Lỗi boundary cục bộ có thể lan thành drift dài.

**Khắc phục**

- Monotonic span DP/shortest path hỗ trợ tập transition cấu hình `m,n ∈ {0..K}`, ít nhất `1↔1`, `1↔2`, `2↔1`, `2↔2`, `1↔3`, `3↔1`, skip.
- Có merge prior, length-ratio prior, punctuation/structure/anchor score và margin.
- Chia block theo anchor, band theo slope cục bộ, detect drift và resync tại anchor kế tiếp.

### S1 — High: chia volume sau alignment gán sai orphan ở boundary

**Bằng chứng code**

- `run_mapping.py:307-353` gán volume theo `han_indices[0]`.
- Dòng chỉ Việt hoặc thiếu index được gán vào `current_vol` (`run_mapping.py:333-345`).

**Tác động**

- Việt đầu volume mới có thể bị gán vào volume trước nếu quanh boundary có skip.
- Một merged pair vắt qua boundary chỉ mang volume của Hán đầu tiên.
- Sau Phase 3, index duplicate/backtrack làm volume attribution càng không đáng tin.

**Khắc phục**

- Cấm align span vượt hard volume boundary.
- Align từng manifest block/volume; không align nhiều volume rồi suy ngược volume từ output.
- Việt-only phải giữ source volume/page của chính nó, không dùng mutable `current_vol`.

### S1 — High: export/merge làm mất kiểm soát chất lượng

**Bằng chứng code**

- `utils/exporters.py:78-122` có ý định xuất full TSV và clean TSV/XLSX, nhưng clean bản phát hành chỉ còn 3 cột, bỏ `similarity_score` và toàn bộ provenance.
- Artifact root thực tế là phép nối Phase 3 full cache, gồm blank, không đi qua clean gate.
- Không có checked-in function tái tạo chính xác root merged artifact; repo chỉ có exporter theo volume.
- `scripts/prepare_data.py:19-38` thu thập mọi file kết thúc `_parallel.tsv`, chỉ drop blank, không deduplicate, không lọc confidence/flags và không phát hiện root merged + per-volume cùng tồn tại.

**Tác động**

- Có thể vừa che lỗi bằng cách drop blank, vừa vô tình phát hành bản full có blank.
- Nếu root merged và per-volume output cùng tồn tại, downstream có nguy cơ đưa cùng cặp vào train hai lần.
- Reviewer không biết pair đến từ page nào, transition gì và tại sao được chấp nhận.

**Khắc phục**

- Ba tập rõ ràng: `accepted`, `review`, `unmatched`; không gọi unmatched là parallel pair.
- Accepted schema phải có source IDs/spans, volume/page, transition `m-n`, component scores, confidence, flags, pipeline fingerprint.
- Exporter phải assert invariant trước khi ghi; merged export phải là một function deterministic được test.

### S1 — High: OCR ordering không tổng quát và không được kiểm chứng ở ingestion

**Bằng chứng code**

- `core/pipeline.py:40-47` và `126-128` sort filename theo chuỗi, không natural sort; `page_10` có thể đứng trước `page_2` nếu không zero-pad.
- `ocr/providers.py:148-244` xoay cố định CCW, group band bằng threshold 90 px và sort band theo geometry cố định; logic này có vẻ chuyên biệt cho một layout/bảng, không có layout classifier hoặc test trên nhiều loại trang.
- `ocr/ensemble.py:19-43` không “vote” theo token/box; chỉ chọn toàn bộ output có nhiều ký tự CJK nhất.
- `run_mapping.py` dùng CSV đã extract nhưng không validate row/page order trước alignment.

**Tác động**

- Chỉ một page/column bị đảo tạo block drift mà semantic aligner khó sửa.
- Output dài hơn có thể được chọn dù lặp cột hoặc sai reading order.

**Khắc phục**

- Natural sort + manifest page order + ID monotonicity validator.
- OCR phải trả structured blocks `{page, bbox, text, confidence, reading_order}`.
- Reading-order test riêng cho vertical columns, mixed header/body và multi-band pages.

### S2 — Medium: cleaning phá hủy thông tin trước khi alignment

**Bằng chứng code**

- `utils/exporters.py:6-46` xóa toàn bộ Hán tự trong câu Việt và mọi parenthetical ngắn ≤15 ký tự.
- `run_mapping.py:181-195` áp dụng cleaner trước khi semantic alignment.

**Tác động**

- Có thể xóa anchor quan trọng, tên riêng, chữ đối chiếu hoặc chú thích thật sự cần cho alignment.
- Không có audit trail từ cleaned text về raw span.

**Khắc phục**

- Luôn giữ `raw_text` và `normalized_text`; scorer chọn view phù hợp.
- Cleaner phải tạo transformation log/offset map và có test fixture.

### S2 — Medium: input/schema anomalies không có gate

Ngoài lỗi swap/missing volume, dữ liệu hiện có còn:

- `q1_sentences.csv` lặp ID đầu `Q1_23_001`.
- `q13_sentences.csv` có sequence `Q13_4_001`, `Q13_4_011`, `Q13_4_002`.
- File tên q8 dùng prefix Q9 và cũng có sequence `Q9_2_001`, `Q9_2_011`, `Q9_2_002`.
- Một số ID Việt dùng page-range dạng `page_11_12`; cần định nghĩa rõ semantics và validator riêng, không nên sort tuple số một cách mù quáng.

Hiện `load_sino_csv()` giữ nguyên order file nhưng bỏ ID khỏi object truyền vào aligner. Mọi anomaly vì thế bị mất dấu sau ingestion.

### S2 — Medium: thiếu test và tài liệu không khớp code/artifact

- Không có unit/regression test cho segmentation, DP, cache, Phase 3 invariant, volume mapping hoặc exporter; chỉ có test OCR/API thủ công.
- `docs/ALIGNMENT_GUIDELINE.md` mô tả “gold corpus”, merge limit và output schema tự tin hơn implementation thực tế.
- README mô tả `main.py` là BERT/m–n dù code dùng mock.

## 6. Các giả thuyết cần xác minh

1. **Đã xác minh q8/q9:** filename và nội dung đúng quyển; prefix ID bị đảo.
2. **Đã xác minh Q11/Q17:** nội dung tồn tại trong file ghép; prefix ID không chuyển theo volume.
3. **Lệch nửa sau corpus chủ yếu bắt đầu tại boundary volume nào?** Tạo anchor gold tại đầu/cuối từng quyển; dự kiến q07_08, q09, q10_11 và q16_17 là điểm lỗi lớn.
4. **Artifact root được tạo bằng bước ngoài repo.** Nội dung chứng minh nó là concat Phase 3 cache, nhưng code merge/renumber root không có trong production script hiện tại; cần tìm notebook/cell hoặc script đã dùng.
5. **CSV Sino có phải OCR reading order đúng không?** ID cho biết page/row nhưng repo chưa có chain tái tạo CSV; cần spot-check ảnh theo các block bất thường.
6. **Score cross-lingual có calibration phù hợp cổ Hán–Việt không?** Threshold 0,32/0,50 hiện là heuristic; cần gold set và calibration curve.
7. **Các parenthetical ngắn có luôn là noise không?** Cần sample stratified trước khi tiếp tục xóa.

## 7. Kiến trúc đề xuất

### 7.1. Ingestion có manifest và provenance

Mỗi unit phải giữ:

```text
source_unit_id
work_id / volume_id / page_id / order_in_page
bbox / reading_order (nếu OCR)
raw_text
normalized_text
boundary_candidates
source_checksum
```

Manifest khai báo mapping Hán–Việt theo volume/page range và expected prefix. Validator chạy trước model:

- file tồn tại, checksum đúng;
- prefix/volume coverage đúng;
- source ID duy nhất;
- page/order đơn điệu theo parser ID đã định nghĩa;
- không thiếu/nhầm block;
- heading/anchor đầu-cuối hợp lý.

### 7.2. Soft segmentation thay hard split

- Hán: punctuation, OCR line break, whitespace, page/column boundary là candidate cut với feature/penalty khác nhau.
- Việt: giữ paragraph/page break; tokenizer chỉ thêm candidate sentence boundary, không xóa cấu trúc.
- Aligner làm việc trên lattice/span liên tiếp, chọn boundary cùng lúc với alignment.
- Hard boundary chỉ dùng cho work/volume và các anchor đã xác minh.

### 7.3. Anchor-first block alignment

Tạo anchor từ:

- tiêu đề quyển/chương/mục;
- chuỗi Hán còn giữ trong bản Việt;
- tên riêng/địa danh, niên hiệu/năm, số lượng/đơn vị;
- page marker hoặc reference ID đã có correspondence;
- rare n-gram/lexicon Hán–Việt.

Chọn chuỗi anchor đơn điệu bằng DP riêng, chia toàn group thành block nhỏ. Nếu anchor conflict, dừng group để review thay vì align xuyên qua.

### 7.4. True monotonic m–n alignment

Mỗi transition `(m,n)` dùng span liên tiếp ở hai phía. Score đề xuất:

```text
S = w_semantic * semantic_score
  + w_anchor   * anchor/entity/numeral_score
  + w_length   * length_ratio_score
  + w_boundary * punctuation/structure_score
  - merge_prior(m,n)
  - skip_prior(m,n)
  - drift_penalty(local_slope)
```

Tập transition nhỏ và có prior: `1-1`, `1-2`, `2-1`, `2-2`, `1-3`, `3-1`, `0-1`, `1-0`. Mở rộng chỉ khi gold data chứng minh cần thiết. Không dùng merge 15/47/74 như lối thoát chung.

DP chạy trong band quanh slope cục bộ giữa hai anchor, không trên full `M×N` không giới hạn. Kết quả phải bảo toàn coverage theo policy: mỗi source unit xuất hiện đúng một lần trong `accepted + review + unmatched`.

### 7.5. Resynchronization

Theo dõi rolling signals:

- nhiều skip liên tiếp;
- confidence/margin thấp;
- slope thay đổi đột ngột;
- length ratio outlier;
- anchor/numeral conflict.

Khi trigger:

1. đóng block hiện tại ở trạng thái review;
2. tìm anchor tin cậy tiếp theo trong cửa sổ mở rộng;
3. restart DP từ anchor đó;
4. không cho lỗi lan qua volume/page boundary.

### 7.6. Confidence và human review

Confidence không chỉ là cosine. Lưu:

- best path score và margin so với path thứ hai;
- component scores;
- transition type và span sizes;
- flags: long merge, skip run, anchor conflict, OCR-order risk, LLM proposal;
- model/config/cache fingerprint.

Routing:

- `accepted`: high confidence, không vi phạm invariant;
- `review`: uncertain/outlier/LLM proposal;
- `unmatched`: một phía, không giả làm parallel pair.

LLM có thể đề xuất trong local window nhưng output phải qua deterministic constrained decoder/validator.

### 7.7. Cache và reproducibility

Mỗi phase ghi manifest:

```text
input_manifest_hash
normalized_units_hash
code_schema_version / git revision
model names + revisions
config hash
phase name
record count + invariant summary
```

Final cache immutable. Checkpoint dùng tên riêng và chỉ resume cùng fingerprint. Rerun cùng input phải cho TSV/JSON deterministic tương đương; thay bất kỳ input/config/model nào phải cache miss.

### 7.8. Export contract

Tối thiểu:

- `*_accepted.tsv/xlsx`
- `*_review.tsv/xlsx`
- `*_unmatched.tsv/xlsx`
- `*_audit.json`

Không bỏ provenance/confidence khỏi workbook review. Merged export phải nhận danh sách volume theo manifest, assert không duplicate `source_unit_id`, renumber pair ID deterministic và ghi summary theo volume.

## 8. Kế hoạch triển khai tuần sau

### Phase 0 — Khóa baseline và sửa data contract (0,5–1 ngày)

- Copy artifact hiện tại thành baseline read-only; không dùng cho train mới.
- Lập gold review set stratified 300–500 vùng: đầu/cuối mỗi volume, mọi block blank dài, vùng q7–q9, q10–11, q16–17 và các outlier độ dài.
- Xác minh/repair manifest q8/q9, Q11, Q17 bằng tiêu đề và scan gốc.
- Chốt schema unit/pair/invariant và acceptance metrics.

Deliverable: manifest hợp lệ + gold set + script fail-fast ordering/coverage.

### Phase 1 — Ingestion/order/provenance (1 ngày)

- Thay load list string bằng structured units có ID/page/volume/raw/normalized.
- Natural sort và parser ID theo từng nguồn.
- Giữ page/column boundary; thêm OCR reading-order fixture.
- Cache fingerprint v1.

Deliverable: normalized unit files deterministic, không chạy model nếu mapping/order sai.

### Phase 2 — Soft segmentation + anchor layer (1–1,5 ngày)

- Xây candidate boundary lattice cho Hán và Việt.
- Extract heading/name/date/numeral anchors.
- Monotonic anchor chain chia group thành local blocks.

Deliverable: visualization/report block boundaries và coverage trước alignment.

### Phase 3 — Monotonic m–n aligner (1,5–2 ngày)

- Implement span transitions nhỏ, merge/skip/length priors và banded DP.
- Bảo toàn source ID coverage và hard volume boundary.
- Benchmark trên gold set so với Phase 1 hiện tại.

Deliverable: path deterministic + metric precision/recall theo transition.

### Phase 4 — Resync, confidence và review queue (1 ngày)

- Drift detector, anchor resync, margin/confidence calibration.
- Nếu dùng LLM, thêm constrained proposal schema + validator; bỏ splice trực tiếp.
- Phân luồng accepted/review/unmatched.

Deliverable: review workbook có provenance, flags và confidence.

### Phase 5 — Export, regression và rerun có kiểm soát (0,5–1 ngày)

- Deterministic merged exporter; dedup gate; audit summary.
- Chạy test suite, rerun từng volume từ cache sạch.
- Spot review theo volume, chỉ publish khi đạt acceptance criteria.

Deliverable: corpus version mới + manifest/fingerprint + báo cáo chênh lệch với baseline.

## 9. Test cases bắt buộc

### 9.1. Unit tests

**Ordering/input**

- `page_1`, `page_2`, `page_10` phải natural-sort đúng.
- q8/q9 phải được nhận diện theo heading/manifest và vẫn giữ original ID để audit.
- q10_11 và q16_17 phải tách được hai volume theo heading; thiếu heading mới là lỗi fail-fast.
- duplicate ID và ID quay lùi phải fail hoặc được ghi explicit exception trong manifest.
- page-range ID như `page_11_12` phải được parser theo spec, không suy đoán.

**Soft segmentation**

- Hán newline không tự động luôn là câu mới.
- Việt page newline không bị xóa mất provenance.
- `1↔2`, `2↔1`, `2↔2`, fragment OCR và câu Việt dài có candidate span đúng.
- Cleaner giữ raw text và transformation map.

**Alignment**

- Perfect `1↔1`.
- Một boundary sai không làm drift phần sau.
- Inject 1 câu thừa/thiếu ở mỗi phía.
- Một block bị swap/missing phải trigger fail/resync, không force-align.
- Không transition nào vượt volume boundary.
- Mỗi source ID xuất hiện đúng một lần trong tổng ba output set.

**Phase 3/LLM**

- Proposal dùng ID trùng, bỏ ID, đảo order, ID ngoài range hoặc plain text phải bị reject.
- Chạy realign hai lần phải không đổi kết quả.
- Chunk size luôn ≤ limit; không cắt xuyên anchor.

**Cache**

- Đổi một ký tự input, threshold, model revision hoặc manifest phải cache miss.
- Checkpoint fingerprint mismatch phải fail rõ ràng.

**Exporter/downstream**

- Accepted không có blank.
- Review/unmatched giữ đủ metadata.
- Root merged + per-volume không được double count.
- Pair ID deterministic, không duplicate, volume order theo manifest.

### 9.2. Regression fixtures từ artifact hiện tại

Tối thiểu lấy fixture quanh các vùng:

- dòng 1–12;
- 499–511;
- 729–894;
- 999–1.011;
- 1.396–1.480;
- 1.791–1.954;
- 2.594–2.630;
- 5.704–5.736;
- 7.611–7.709;
- 7.999–8.011;
- 10.239–10.290;
- 11.999–12.011;
- 15.246–15.267.

Fixture phải gắn source volume/page/ID sau khi repair manifest, không chỉ dùng row number của workbook.

## 10. Tiêu chí nghiệm thu

### Invariant bắt buộc (pass/fail)

- 100% group pass expected volume/prefix/page coverage trước model.
- 0 source ID duplicate ngoài trường hợp manifest đánh dấu và giải thích.
- Path monotonic; 0 backtrack, 0 reuse index.
- Mỗi Hán/Viet source unit xuất hiện đúng một lần trong `accepted + review + unmatched`.
- 0 blank trong accepted corpus.
- 0 pair vượt hard volume boundary.
- Cache fingerprint đầy đủ và rerun idempotent.
- Merged artifact được tái tạo bằng checked-in command/test, không bằng bước thủ công.

### Chất lượng trên gold set

- Precision của `accepted` ≥95% ở mức pair/span exact hoặc semantic-equivalent theo rubric đã chốt.
- Recall coverage của gold links ≥90%; phần chưa chắc chắn phải vào review, không force-align.
- Boundary accuracy cho `1↔2`/`2↔1` ≥90% trên subset tương ứng.
- Sau injected boundary disturbance, path resync trong tối đa 5 source units hoặc tại anchor gần nhất.
- High-confidence bucket có error rate ≤5% qua manual review stratified.

### Vận hành

- Báo cáo theo volume: input counts, accepted/review/unmatched, transition distribution, long-merge/skip-run flags, confidence calibration.
- Không cho train pipeline đọc artifact nếu audit manifest/invariant chưa pass.

## 11. Rủi ro và cách giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Gold set thiên lệch vùng dễ | Stratify theo volume, boundary type, OCR quality, blank run và score bucket |
| Anchor sai do OCR | Dùng nhiều tín hiệu, monotonic consistency và manual review anchor conflict |
| Embedding cổ Hán–Việt calibration kém | Benchmark nhiều scorer trên gold; giữ anchor/length/structure score độc lập |
| Soft segmentation làm state space lớn | Hard volume block, anchor partition, banded DP, transition set nhỏ |
| LLM không deterministic/vi phạm schema | Chỉ proposal theo ID, temperature 0, strict validator, invalid → review |
| Mất dữ liệu do cleaner | Raw immutable + normalized view + transformation log |
| Cache cũ nhiễm output mới | Namespace theo fingerprint; không overwrite final cache |
| Review queue quá lớn | Calibrate confidence, ưu tiên block conflict/long merge/blank run/outlier |

## 12. Những việc không nên làm

- Không chỉ tăng `max_merge_han`, đổi threshold hoặc skip penalty để làm match rate đẹp hơn.
- Không drop blank rồi gọi phần còn lại là “gold corpus” nếu chưa kiểm tra drift/duplicate/semantic.
- Không dùng LLM tự do để rewrite/splice path mà không có coverage/order validator.
- Không tái dùng cache chỉ vì filename giống nhau.
- Không align toàn bộ nhiều volume trong một global DP rồi suy ngược volume bằng `current_vol`.
- Không hard-split Hán theo mọi whitespace/newline rồi trông chờ merge lớn sửa lại.
- Không xóa Hán tự/chú thích khỏi bản Việt trước khi lưu raw và provenance.
- Không tin mapping theo filename khi ID/heading không khớp.
- Không phát hành workbook chỉ có ba cột nếu reviewer cần confidence, source IDs và flags.
- Không đưa artifact HVB_001 hiện tại vào vòng train mới trước khi repair manifest và rerun.

## 13. Thứ tự ưu tiên đề xuất

1. Dừng dùng artifact hiện tại cho train/publish.
2. Chuẩn hóa metadata q8/q9 và tách Q11/Q17 theo heading có fail-fast.
3. Vô hiệu hóa mock `BERTAlignerWrapper` khỏi đường production.
4. Thêm source-unit provenance và cache fingerprint.
5. Thay Phase 3 splice bằng constrained validator; thêm invariant test.
6. Implement anchor-partitioned monotonic m–n alignment + resync.
7. Xuất accepted/review/unmatched có confidence và audit manifest.
8. Rerun từng volume từ cache sạch, review theo gold set rồi mới merge.
