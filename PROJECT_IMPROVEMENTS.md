# BÁO CÁO TỔNG KẾT NÂNG CẤP VÀ CẢI TIẾN DỰ ÁN NGHIÊN CỨU KHOA HỌC
**Đề tài:** Phân loại chọn lọc và Hiệu chuẩn xác suất trong Nhận diện Căng thẳng Đáng tin cậy từ Tín hiệu Sinh lý Đeo Cổ tay  
**Tập dữ liệu:** WESAD (Wearable Stress and Affect Detection) - Thiết bị Empatica E4  
**Nhóm tác giả:** 
- Nguyễn Minh Bảo (`baonguyenminh17@gmail.com`)
- Khưu Lê Trúc Nhi (`nhiklt2006@gmail.com`)
- Phan Tấn Đạt (`ptundatwork@gmail.com`)
- Trương Quốc Huy (`huy522006@gmail.com`)

---

## 1. TỔNG QUAN VÀ ĐỘNG LỰC CẢI TIẾN

Dự án ban đầu gặp một số hạn chế phổ biến trong các nghiên cứu ứng dụng học máy:
1. **Câu hỏi nghiên cứu (RQs) chưa sắc bén:** Các câu hỏi trước đây nặng tính so sánh kỹ thuật thông thường ("Mô hình nào tốt hơn?") mà thiếu đi luận điểm khoa học thực chứng và giá trị đóng góp học thuật (novel contributions).
2. **Nguy cơ rò rỉ dữ liệu (Data Leakage):** Việc trích xuất đặc trưng với cửa sổ trượt (sliding window 60 giây, bước nhảy 5 giây) tạo ra độ trùng lặp lên tới 91.7% giữa các mẫu liên tiếp. Nếu hiệu chuẩn xác suất (Probability Calibration) hoặc phân chia tập train/val không cô lập theo từng đối tượng (`subject_id`), kết quả báo cáo sẽ bị thổi phồng giả tạo (optimistic bias).
3. **Số lượng mô hình thử nghiệm còn ít:** Việc chỉ đánh giá trên 1-2 mô hình khiến bài báo dễ bị nghi ngờ là "thiên vị thuật toán" (algorithm bias) hoặc "ăn may tham số", làm giảm độ uy tín khi nộp bài cho các hội thảo/tạp chí uy tín (IEEE/ACM).
4. **Hiện tượng báo động giả do chuyển động thể chất (Motion Artifacts):** Các cảm biến đeo cổ tay (Empatica E4) cực kỳ nhạy cảm với chuyển động cơ học (bước đi, rung lắc tay), khiến tín hiệu gia tốc (ACC) và dẫn truyền da (EDA) tăng đột biến tương tự như khi bị căng thẳng tâm lý (stress), gây ra tỷ lệ báo động giả rất cao ($\text{FAR}_{\text{motion}}$).
5. **Hình ảnh và cấu trúc thư mục chưa chuẩn mực:** Các biểu đồ còn chứa nhãn tiếng Việt, font chữ và độ phân giải chưa đạt chuẩn IEEE; mã nguồn còn lưu nhiều file dạng "Bản sao của..." tạo cảm giác rời rạc, thiếu tính chuyên nghiệp.

Dự án đã được tái cấu trúc và nâng cấp toàn diện lên **Version 3** với chất lượng đạt chuẩn nghiên cứu khoa học quốc tế.

---

## 2. BỐN CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS) CHUẨN KHOA HỌC

Để bài báo có tính học thuật cao và định vị rõ đóng góp, nhóm đã tái định hình thành 4 câu hỏi nghiên cứu cụ thể:

* **RQ1 (Lỗ hổng Báo động giả do Vận động):** *Mô hình nhận diện căng thẳng truyền thống triển khai trên thiết bị đeo cổ tay dễ bị tổn thương bởi các nhiễu động vận động (motion artifacts) đến mức độ nào khi đánh giá trong điều kiện vận động cường độ cao?*
* **RQ2 (Hiệu chuẩn Xác suất Có tính đến Cấu trúc Nhóm):** *Phương pháp hiệu chuẩn hậu xử lý (Platt Scaling và Isotonic Regression) được tối ưu hóa bằng phân chia chéo theo nhóm đối tượng (Group-aware CV) cải thiện độ tin cậy của xác suất dự báo (giảm sai số ECE và Brier Score) như thế nào trên dữ liệu đối tượng hoàn toàn chưa thấy?*
* **RQ3 (Cơ chế Phân loại Chọn lọc và Triệt tiêu Báo động giả):** *Việc kết hợp ngưỡng tin cậy xác suất $\tau$ và cơ chế từ chối dựa trên cảm biến gia tốc $\theta_{\text{motion}}$ triệt tiêu tỷ lệ báo động giả trong điều kiện vận động ($\text{FAR}_{\text{motion}}$) hiệu quả ra sao, và sự đánh đổi với tỷ lệ bao phủ (coverage) là gì?*
* **RQ4 (Độ bền vững & Giải thích Định lượng Sinh lý học):** *Mô hình có giữ được tính khái quát hóa và độ ổn định qua đa hạt giống ngẫu nhiên (multi-seed ablation) không, và cơ chế TreeSHAP giải thích đóng góp của các kênh tín hiệu sinh lý (EDA vs ACC vs TEMP vs BVP) như thế nào đối với các ca thất bại sinh lý học cá nhân (như S12)?*

---

## 3. KHẮC PHỤC TRIỆT ĐỂ RÒ RỈ DỮ LIỆU BẰNG DUAL-LEVEL LOSO

Quy trình đánh giá Leave-One-Subject-Out (LOSO) được thiết kế lại theo cấu trúc **Dual-Level (Hai tầng bảo vệ)**:
1. **Tầng ngoài (Outer LOSO - 15 Folds):**
   - Ở mỗi fold $k \in \{1, \dots, 15\}$, toàn bộ dữ liệu của đối tượng $S_k$ được tách riêng hoàn toàn làm tập kiểm thử (Test Subject).
   - 14 đối tượng còn lại đóng vai trò là tập huấn luyện (Training Subjects).
2. **Tầng trong (Inner Group-Aware Calibration):**
   - Khắc phục lỗi rò rỉ cửa sổ trượt: Tuyệt đối không dùng k-fold ngẫu nhiên.
   - Thay vào đó, áp dụng `GroupKFold(n_splits=3)` gom nhóm theo `subject_id` chỉ trên 14 đối tượng huấn luyện để vừa huấn luyện mô hình cơ sở, vừa điều chỉnh bộ hiệu chuẩn Isotonic/Sigmoid.
   - Khi mô hình được đánh giá trên $S_k$, đối tượng này chưa từng xuất hiện ở bất kỳ bước chuẩn hóa (scaler), huấn luyện hay hiệu chuẩn nào.

---

## 4. BENCHMARK ĐA MÔ HÌNH VÀ TINH CHỈNH TỐI ƯU (7 MODEL BENCHMARK)

### 4.1. Tại sao phải chạy 7 mô hình?
Để chứng minh rằng **"sự nhạy cảm với chuyển động và độ thiếu tin cậy xác suất là vấn đề cố hữu của dữ liệu sinh lý học đeo tay, chứ không phải do một thuật toán cụ thể"**, nghiên cứu đã benchmark 7 mô hình đại diện cho 5 họ thuật toán kinh điển:
1. *Linear Models:* Logistic Regression (L2 penalty)
2. *Decision Trees:* Decision Tree (CART, max_depth=6)
3. *Instance-based:* k-Nearest Neighbors ($k=11$, Euclidean distance)
4. *Ensemble Boosting:* AdaBoost (SAMME.R, 100 estimators)
5. *Ensemble Bagging:* Random Forest (150 trees, max_depth=10, balanced subsample)
6. *Gradient Boosting:* LightGBM (100 trees, learning_rate=0.05, num_leaves=31)
7. *Extreme Gradient Boosting:* XGBoost (200 trees, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8)

### 4.2. Kết quả thực nghiệm đối sánh (15-Fold LOSO Trên 5,294 Cửa sổ)

| Mô hình (Model Family) | Macro $F_1$ Đầy đủ (100% Coverage) | $\text{FAR}_{\text{motion}}$ Ban đầu | Macro $F_1$ Chọn lọc ($\tau=0.80$) | $\text{FAR}_{\text{motion}}$ Chọn lọc | Mức giảm Báo động giả | Tỷ lệ Bao phủ (Coverage) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 71.39% | 23.05% | 76.54% | 10.37% | **-55.0%** | 60.77% |
| **Decision Tree** | 75.14% | 17.51% | 80.48% | 7.74% | **-55.8%** | 68.21% |
| **k-Nearest Neighbors** | 77.82% | 15.34% | 85.06% | 3.52% | **-77.1%** | 62.45% |
| **AdaBoost** | 81.16% | 14.73% | 89.20% | 2.89% | **-80.4%** | 67.12% |
| **LightGBM** | 86.41% | 13.06% | 93.85% | 2.05% | **-84.3%** | 71.05% |
| **Random Forest** | 85.90% | 13.88% | 93.10% | 2.18% | **-84.3%** | 70.82% |
| **XGBoost (Đề xuất)** | **88.24%** | **12.20%** | **95.15%** | **1.69%** | **-86.1%** | **69.38%** |

**Kết luận thực nghiệm:**
- Khi ép mô hình dự đoán 100% (không có phân loại chọn lọc), tất cả 7 mô hình đều gặp tỷ lệ báo động giả cao (từ 12.20% đến 23.05%).
- Cơ chế Phân loại Chọn lọc ($\tau=0.80$) giúp triệt tiêu báo động giả đồng loạt trên cả 7 mô hình (từ 55.0% đến 86.1%).
- XGBoost sau khi được tinh chỉnh tham số và hiệu chuẩn xác suất đạt hiệu năng vượt trội nhất ($F_1 = 95.15\%$, $\text{FAR}_{\text{motion}} = 1.69\%$).

---

## 5. CHUẨN HÓA TOÀN BỘ HÌNH ẢNH 100% TIẾNG ANH (CHUẨN IEEE TRANSACTIONS)

Toàn bộ 7 hình ảnh trong bài báo đã được thiết kế lại ở độ phân giải 300 DPI, phông chữ đồng nhất, nhãn trục, tiêu đề, chú thích 100% bằng tiếng Anh học thuật:

1. `fig_multimodel_benchmark.png`: Biểu đồ so sánh 7 mô hình về mức độ triệt tiêu $\text{FAR}_{\text{motion}}$ và độ tăng trưởng Macro $F_1$.
2. `far_motion_reduction.png`: So sánh trực quan tỷ lệ báo động giả trước và sau khi áp dụng Selective Classification trên các mô hình.
3. `calibration_curve.png`: Reliability Diagram và phân phối độ tin cậy giữa Uncalibrated vs Calibrated (Platt/Isotonic).
4. `risk_coverage_curve.png`: Đường cong đánh đổi giữa rủi ro (Risk / Error Rate) và mức độ bao phủ (Coverage) theo các ngưỡng $\tau \in [0.50, 0.95]$.
5. `ablation_acc_impact.png`: Nghiên cứu cắt bỏ kênh ACC qua 5 hạt giống ngẫu nhiên (Seeds: 42, 100, 2024, 7, 999) chứng minh ACC cải thiện $F_1$ thêm $+8.43\%$ và giảm báo động giả $-7.76\%$.
6. `shap_summary.png`: Phân tích gán thuộc tính toàn cục TreeSHAP trên 25 đặc trưng (EDA tonic/phasic, ACC peak/energy, HR mean, TEMP slope).
7. `fig_failure_S12.png`: Phân tích ca thất bại đối tượng S12 (Hiện tượng suy giảm đáp ứng điện da sinh lý - Electrodermal Hypo-responder dù nhịp tim tăng vọt do stress).

Tất cả các hình ảnh được lưu trữ tập trung duy nhất tại:
- `Latex/Images/` (giúp thư mục `Latex/` hoàn toàn khép kín, sẵn sàng nén zip nộp bài hoặc đẩy lên Overleaf)

---

## 6. PIPELINE TRIỂN KHAI VÀ LƯU TRỮ MÔ HÌNH (PRODUCTION DEPLOYMENT)

Mô hình không chỉ dừng lại ở kết quả học thuật mà đã được đóng gói thành các tệp nhị phân và mã nguồn sẵn sàng đưa vào sản phẩm thực tế:
- `Models/trustworthy_xgb_pipeline.joblib`: Pipeline hoàn chỉnh gồm `StandardScaler` + `CalibratedClassifierCV(XGBoost)`.
- `Models/trustworthy_rf_pipeline.joblib`: Pipeline Random Forest dự phòng.
- `Models/xgboost_base_model.json`: Định dạng JSON nguyên bản của XGBoost, cho phép tích hợp trực tiếp vào ứng dụng di động (Android / iOS / CoreML), thiết bị nhúng C/C++ hoặc vi điều khiển edge computing mà không cần thư viện Python.
- `Models/config_inference.json`: Cấu hình sản xuất lưu trữ chính xác thứ tự 25 đặc trưng đầu vào, thông số trung bình/độ lệch chuẩn của Scaler, các ngưỡng quyết định ($\tau = 0.80$, $\theta_{\text{motion}} = 15.385$), và nhãn giải mã.
- `Models/predict.py`: Lớp triển khai `TrustworthyStressPredictor` được thiết kế theo chuẩn hướng đối tượng (OOP), có cơ chế kiểm tra chuyển động (motion filter), hiệu chuẩn xác suất và từ chối dự đoán khi không chắc chắn (rejection/abstention). Đã được kiểm thử và xác nhận chạy chính xác.

---

## 7. CẤU TRÚC THƯ MỤC VÀ HỆ THỐNG NOTEBOOK CHUẨN MỰC

Hệ thống thư mục đã được tinh gọn, xóa bỏ hoàn toàn các thư mục trùng lặp (`Nhi/`, `Bao/`), các file tạm compilation (`.aux`, `.log`, `.out`), di chuyển các bảng kết quả sang `Results/`, và chỉ giữ lại duy nhất thư mục hình ảnh trong `Latex/Images/`. Toàn bộ các notebook trong `Notebook/` được quy chuẩn đồng bộ 100% theo phong cách đặt tên của nhóm (`dsr301-stress-detection-...`) và đều đã được chạy lưu sẵn toàn bộ kết quả thực thi (outputs) chi tiết:

```text
Project/
├── Dataset/
│   ├── wesad_wrist_features.csv             <-- 5,294 cửa sổ quan sát x 25 đặc trưng sinh lý
│   └── experiment_results_summary.json
├── Latex/
│   ├── Images/                              <-- Toàn bộ 7 biểu đồ chuẩn tiếng Anh 300 DPI
│   ├── version 1.tex
│   ├── version 2.tex / version 2.pdf
│   └── version 3.tex / version 3.pdf        <-- Phiên bản cập nhật đầy đủ nhất
├── Models/
│   ├── config_inference.json                <-- Metadata & tham số suy luận
│   ├── predict.py                           <-- Module suy luận kiểm thử mẫu
│   ├── trustworthy_xgb_pipeline.joblib      <-- Pipeline XGBoost sản xuất
│   ├── trustworthy_rf_pipeline.joblib       <-- Pipeline Random Forest sản xuất
│   └── xgboost_base_model.json              <-- Model JSON đa nền tảng
├── Notebook/
│   ├── dsr301-stress-detection-version-2.ipynb              <-- Thu thập dữ liệu từ Kaggle (giữ nguyên gốc)
│   ├── dsr301-stress-detection-modeling-version-1.ipynb     <-- Mô hình hóa sơ bộ baseline (đã chạy lưu output)
│   ├── dsr301-stress-detection-modeling-version-2.ipynb     <-- Pipeline thống nhất Benchmark 7 mô hình (đã chạy lưu output)
│   ├── dsr301-stress-detection-loso-evaluation.ipynb        <-- Đánh giá chuyên sâu LOSO 15-folds (đã chạy lưu output)
│   ├── dsr301-stress-detection-ablation-study.ipynb         <-- Nghiên cứu cắt bỏ ACC trên 5 seeds (đã chạy lưu output)
│   └── dsr301-stress-detection-shap-interpretability.ipynb  <-- Giải thích TreeSHAP & ca S12 (đã chạy lưu output)
├── Results/                                 <-- Lưu trữ các bảng số liệu xuất ra (LOSO, Ablation, SHAP)
├── Papers/
│   ├── version 2.pdf
│   └── version 3.pdf                        <-- Bản PDF hoàn chỉnh
├── PROJECT_IMPROVEMENTS.md                  <-- Báo cáo chi tiết cải tiến dự án
├── Presentation/
├── Slide/
├── Workflow/
└── Chat/
```

*Lưu ý:* Notebook gốc thu thập dữ liệu từ Kaggle (`dsr301-stress-detection-version-2.ipynb`) được giữ nguyên vẹn 100% không chạy lại theo đúng yêu cầu để bảo toàn dữ liệu gốc. Tất cả các notebook phân tích và mô hình hóa còn lại đều đã được thực thi và lưu đầy đủ kết quả đầu ra trực quan (bảng biểu, log, biểu đồ).

---

## 8. CẬP NHẬT TÀI LIỆU LATEX (VERSION 3)

Tệp [version 3.tex](file:///h:/My%20Drive/FPTU/S5/DSR301m/Project/Latex/version%203.tex) và [version 3.pdf](file:///h:/My%20Drive/FPTU/S5/DSR301m/Project/Latex/version%203.pdf) đã hoàn thiện:
1. Thông tin liên hệ và email của 4 thành viên trong nhóm đã được sửa đúng định dạng.
2. Đưa đầy đủ 4 Câu hỏi nghiên cứu (RQs) vào phần Introduction.
3. Bổ sung **Table I (Comprehensive 7-Model Benchmark)** chi tiết về cả 7 họ mô hình.
4. Bổ sung **Table II (Risk-Coverage Trade-off Curve Analysis)** phân tích các ngưỡng $\tau \in [0.50, 0.95]$.
5. Bổ sung **Table III (Multi-seed Ablation Study)** qua 5 hạt giống độc lập.
6. Cập nhật tiểu mục phân tích ca thất bại S12 (Atypical Physiological Hypo-responder).
7. Biên dịch thành công tệp PDF sắc nét, sẵn sàng nộp báo cáo hoặc gửi lên Overleaf.
