# Tài nguyên Dự án - Hệ thống Gợi ý Sản phẩm Mỹ phẩm

## 📁 Liên kết Google Drive

### 1. Hệ thống MLOps (Backend)
🔗 **Link:** [https://drive.google.com/drive/folders/1O7zOjDmeI3UIuzDWgUqPdxolqvA22dCk?usp=sharing](https://drive.google.com/drive/folders/1O7zOjDmeI3UIuzDWgUqPdxolqvA22dCk?usp=sharing)

**Nội dung bao gồm:**
- Code hệ thống MLOps
- Pipeline huấn luyện mô hình
- Hệ thống serving và API
- **Dữ liệu (Data)** - xem hướng dẫn bên dưới

### 2. Code Web (Frontend)
🔗 **Link:** [https://drive.google.com/drive/folders/1A85Q9E4Se1fnG5RKAMFdP1neGThtrD_Y?usp=sharing](https://drive.google.com/drive/folders/1A85Q9E4Se1fnG5RKAMFdP1neGThtrD_Y?usp=sharing)

**Nội dung bao gồm:**
- Source code giao diện web
- Frontend application

---

## 📊 Hướng dẫn sử dụng Data

### Vị trí dữ liệu
Dữ liệu nằm trong thư mục **Hệ thống MLOps** với đường dẫn:

```
data/published_data/
```

### Cách tải và sử dụng

1. **Tải thư mục MLOps** từ Google Drive
2. **Giải nén** (nếu cần)
3. **Copy thư mục `data/published_data/`** vào project của bạn
4. Đảm bảo cấu trúc thư mục như sau:

```
viecomrec/
├── data/
│   ├── published_data/          ← Dữ liệu gốc
│   │   ├── data_reviews_purchase.csv
│   │   ├── data_product.csv
│   │   └── data_product_attribute.csv
│   └── processed/               ← Dữ liệu đã xử lý (tự động tạo)
├── recsys/
├── service/
└── ...
```

### Các file dữ liệu chính

| File | Mô tả |
|------|-------|
| `data_reviews_purchase.csv` | Dữ liệu đánh giá và mua hàng của người dùng |
| `data_product.csv` | Thông tin sản phẩm (tên, mô tả, giá, brand...) |
| `data_product_attribute.csv` | Thuộc tính sản phẩm (thành phần, công dụng, loại da...) |

---

## ⚠️ Lưu ý quan trọng

1. **Encoding:** Tất cả file CSV sử dụng encoding `UTF-8` (hỗ trợ tiếng Việt)
2. **Kích thước:** Dữ liệu gồm ~300K users, 2.2K products, 369K interactions
3. **Quyền truy cập:** Đảm bảo bạn đã được cấp quyền truy cập Google Drive

---

## 🔧 Cài đặt nhanh

```bash
# Clone repository
git clone https://github.com/viecomrec

# Tải data từ Google Drive và đặt vào data/published_data/

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy pipeline xử lý dữ liệu
python scripts/run_task01_complete.py
```

---

## 📞 Liên hệ

Nếu gặp vấn đề về quyền truy cập hoặc dữ liệu, vui lòng liên hệ:
- **GitHub:** [https://github.com/viecomrec](https://github.com/viecomrec)
- **Email:** [Liên hệ qua GitHub Issues]
