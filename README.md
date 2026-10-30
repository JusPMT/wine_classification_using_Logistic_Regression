# ĐỒ ÁN MÔN HỌC MÁY (MACHINE LEARNING)
## Đề tài 17: Phân loại rượu bằng Logistic Regression (Wine Classification using Logistic Regression)

---

### 1. Tổng quan đề tài

- **Học phần**: Machine Learning (Thời lượng: 12 tuần)
- **Dạng bài toán**: Phân lớp dữ liệu (Classification)
- **Mô hình chính**: Hồi quy Logistic (Logistic Regression)
- **Bài toán cụ thể**: dự đoán một mẫu rượu vang đỏ Vinho Verde là **đạt chuẩn** (`quality >= 6`, nhãn 1) hay **chưa đạt** (`quality < 6`, nhãn 0) từ 11 chỉ số hóa lý. Dữ liệu: UCI Wine Quality, tập rượu vang đỏ (1.599 mẫu).
- **Mục tiêu**:
  - Triển khai toàn diện quy trình Học máy (Machine Learning Pipeline) từ tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện đến đánh giá mô hình.
  - Cài đặt mô hình Logistic Regression từ đầu (From Scratch) sử dụng thuần toán học và thư viện nền tảng NumPy (hàm Sigmoid, hàm mất mát Cross-Entropy, Gradient Descent, điều chuẩn L2).
  - Cài đặt mô hình đối chuẩn (Benchmark) sử dụng thư viện chuyên dụng `scikit-learn`.
  - Thực nghiệm, so sánh độ chính xác và hiệu năng giữa hai phương pháp tiếp cận dựa trên các chỉ số: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
  - Hoàn thiện báo cáo khoa học theo chuẩn biểu mẫu `MauBaoCaoHocMay.docx` (tuân thủ quy chuẩn trích dẫn IEEE) và tài liệu trình chiếu phục vụ báo cáo tiến độ (Tuần 11) và vấn đáp kết thúc môn (Tuần 12).

---

### 2. Cấu trúc thư mục dự án

```text
wine_classification_using_Logistic_Regression/
│
├── .gitignore                      # Danh sách các tệp và thư mục loại trừ khỏi Git tracking
├── README.md                       # Tài liệu hướng dẫn và tổng quan dự án
├── requirements.txt                # Danh sách thư viện Python cần thiết
│
├── data/                           # Quản lý tập dữ liệu
│   ├── raw/                        # Dữ liệu gốc winequality-red.csv (UCI Wine Quality)
│   └── processed/                  # Dữ liệu sau tiền xử lý, làm sạch và chuẩn hóa
│
├── notebooks/                      # Jupyter Notebooks phục vụ khám phá và thử nghiệm
│   ├── 01_eda.ipynb                # Khám phá và phân tích trực quan dữ liệu (EDA)
│   ├── 02_model_scratch.ipynb      # Cài đặt và thực nghiệm mô hình Logistic Regression từ đầu
│   ├── 03_model_sklearn.ipynb      # Cài đặt mô hình bằng thư viện Scikit-learn và so sánh
│   └── 04_evaluation.ipynb         # Tuần 9–10: đánh giá, tinh chỉnh siêu tham số, phân tích lỗi
│
├── src/                            # Mã nguồn chính của dự án (dạng module chuẩn hóa)
│   ├── __init__.py
│   ├── data/                       # Tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── make_dataset.py         # Thu thập và tải tập dữ liệu
│   │   └── preprocess.py           # Làm sạch, tạo nhãn, chia train/test, chuẩn hóa (StandardScaler)
│   ├── models/                     # Thuật toán và mô hình
│   │   ├── __init__.py
│   │   ├── logistic_regression_scratch.py  # Thuật toán tự cài đặt bằng NumPy
│   │   ├── logistic_regression_sklearn.py  # Huấn luyện mô hình Scikit-learn
│   │   └── evaluate.py             # Hàm tính toán ma trận nhầm lẫn và các chỉ số đo lường
│   └── utils/                      # Công cụ tiện ích
│       ├── __init__.py
│       └── visualization.py        # Vẽ đồ thị phân phối, heatmap, learning curves
│
├── reports/                        # Báo cáo kết quả và tài liệu trình bày
│   ├── MauBaoCaoHocMay.docx        # Biểu mẫu báo cáo chuẩn môn học
│   ├── figures/                    # Biểu đồ và hình ảnh trích xuất phục vụ bài báo cáo
│   └── presentations/              # Slide trình bày tiến độ (Tuần 11) và vấn đáp (Tuần 12)
│
└── saved_models/                   # Lưu trữ trọng số mô hình hoặc đối tượng sau huấn luyện
```

---

### 3. Quy trình thực hiện (Machine Learning Pipeline)

1. **Khám phá dữ liệu (EDA - Exploratory Data Analysis)**:
   - Thống kê mô tả các đặc tính hóa lý của rượu (nồng độ cồn, các loại axit, đường dư, SO₂, sulphate,...).
   - Kiểm tra tương quan giữa các biến độc lập và nhãn phân loại (Correlation Matrix).
   - Phát hiện các giá trị ngoại lai (outliers) hoặc giá trị thiếu sót (missing values).

2. **Tiền xử lý và chuẩn bị dữ liệu**:
   - Xử lý phân phối lệch, loại bỏ hoặc hiệu chỉnh ngoại lai.
   - Chuẩn hóa thang đo đặc trưng (Feature Scaling: Standardization / Normalization) do Logistic Regression nhạy cảm với độ chênh lệch biên độ giá trị.
   - Phân chia tập dữ liệu thành Train / Validation / Test theo tỷ lệ hợp lý (ví dụ: 70/15/15 hoặc 80/20 kèm Stratified K-Fold).

3. **Xây dựng mô hình**:
   - **Mô hình tự cài đặt (From Scratch)**: Triển khai từ cơ sở toán học với thuật toán tối ưu Gradient Descent, hàm kích hoạt Sigmoid (bài toán nhị phân), kỹ thuật điều chuẩn (L2 Regularization).
   - **Mô hình Scikit-Learn**: Cấu hình bộ siêu tham số tối ưu thông qua Grid Search / Cross-Validation.

4. **Thực nghiệm và đánh giá**:
   - So sánh đường hội tụ (Loss convergence curve) và thời gian huấn luyện.
   - Đánh giá tổng hợp bằng Confusion Matrix, Precision, Recall, F1-Score và ROC-AUC.

5. **Tổng kết và đóng gói**:
   - Biên soạn báo cáo hoàn chỉnh định dạng PDF theo mẫu `MauBaoCaoHocMay.docx`.
   - Chuẩn bị slide báo cáo ngắn gọn, trực quan, phục vụ thuyết trình 10 phút và vấn đáp 15 phút.

---

### 4. Hướng dẫn thiết lập môi trường

#### Yêu cầu hệ thống
- Python 3.10 trở lên.
- Git.

#### Các bước cài đặt

1. **Tạo và kích hoạt môi trường ảo**:
   ```bash
   python -m venv venv
   # Trên Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # Trên Linux/macOS:
   source venv/bin/activate
   ```

2. **Cài đặt các gói thư viện phụ thuộc**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Chạy kiểm thử dữ liệu hoặc khởi chạy Notebook**:
   ```bash
   jupyter lab
   # hoặc
   jupyter notebook
   ```

---

### 5. Cách chạy mã nguồn

```bash
# (Tùy chọn) tải lại dữ liệu gốc từ UCI; tập tin đã có sẵn trong data/raw/
python -m src.data.make_dataset

# Tiền xử lý: loại 240 bản ghi trùng, tạo nhãn, chia 80/20 có phân tầng, chuẩn hóa z-score
python -m src.data.preprocess            # thêm --keep-duplicates để giữ bản ghi trùng

# Chạy lần lượt bốn notebook trong notebooks/
jupyter notebook
```

| Notebook | Nội dung | Mục trong báo cáo |
|---|---|---|
| `01_eda.ipynb` | Thống kê, phân bố điểm, boxplot, tương quan, ngoại lai | 3.1 – 3.2 |
| `02_model_scratch.ipynb` | Mô hình tự cài đặt, khảo sát tốc độ học, đánh giá | 4.4 |
| `03_model_sklearn.ipynb` | GridSearchCV, đối chuẩn với bản tự cài đặt, so sánh thời gian | 4.5 |
| `04_evaluation.ipynb` | Kiểm tra cuối, kiểm định chéo, C/L1/L2, ngưỡng, bootstrap hệ số, phân tích lỗi | 5.1–5.2 |

Quy ước kỹ thuật: nhãn `y = 1` nếu `quality >= 6`; `train_test_split(test_size=0.2, stratify=y, random_state=42)`; `StandardScaler` chỉ `fit` trên tập train; `lam = 1 / C` (bản tự cài đặt `lam = 1` tương ứng scikit-learn `C = 1`); hệ số chặn `b` không bị điều chuẩn. Hình xuất ra nằm trong `reports/figures/`, mô hình đã huấn luyện nằm trong `saved_models/`.

---

### 6. Lịch trình và mốc kiểm soát (Milestones)

- **Tuần 2 - 4**: Thiết lập môi trường, thu thập dữ liệu, phân tích EDA và hoàn thành chương 1, 2 của báo cáo.
- **Tuần 5 - 8**: Cài đặt mô hình tự code (scratch), mô hình thư viện (scikit-learn), hoàn thiện tiền xử lý và chương 3, 4.
- **Tuần 9 - 10**: Thực nghiệm đánh giá, phân tích lỗi, tinh chỉnh siêu tham số và hoàn thiện chương 5, 6.
- **Tuần 11**: Báo cáo tiến độ đồ án (Thời lượng trình bày: tối đa 10 phút).
- **Tuần 12**: Vấn đáp cuối kỳ (Thời lượng: tối đa 15 phút), nộp mã nguồn và báo cáo hoàn chỉnh.
