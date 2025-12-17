# Hệ thống Gợi ý Sữa rửa mặt

> Đồ án cuối kỳ học phần **Nhập môn Trí tuệ Nhân tạo (MAT1206E)** tại Trường Đại học Khoa học Tự nhiên, ĐHQGHN (VNU-HUS).

---

## ℹ️ Thông tin Dự án

| Danh mục | Chi tiết |
| :--- | :--- |
|**Tên dự án** | Hệ thống Gợi ý Sữa rửa mặt |
| **Học phần** | MAT1206E – Nhập môn Trí tuệ Nhân tạo |
| **Học kỳ** | Học kỳ 1, Năm học 2025 – 2026 |
| **Trường** | VNU-HUS (Đại học Quốc gia Hà Nội – Trường ĐHKHTN) |
| **Giảng viên** | ThS. Hoàng Anh Đức |
| **Ngày nộp** | 30/11/2025 |

## 📚 Tài liệu & Tài nguyên

Bạn có thể tham khảo tài liệu chi tiết của dự án tại các liên kết dưới đây:

- 📄 **Báo cáo dự án (PDF):** [Xem Báo cáo (IAI.pdf)](https://github.com/Hoang-k68a3hus/project_IAI/blob/main/IAI.pdf)
- 📊 **Slide thuyết trình:** [Xem Slide (SL.pdf)](https://github.com/Hoang-k68a3hus/project_IAI/blob/main/SL.pdf)
- 🔗 **Kho mã nguồn:** [github.com/Hoang-k68a3hus/project_IAI](https://github.com/Hoang-k68a3hus/project_IAI)

## 👥 Thành viên nhóm & Phân công

| STT | Họ tên | Mã sinh viên | GitHub | Đóng góp chính |
| :-: | :--- | :---: | :--- | :--- |
| 1 | **Mai Huy Hoàng** | 23001878 | [@Hoang-k68a3hus](https://github.com/Hoang-k68a3hus) | Phát triển hệ thống, xử lý data, code hệ thống |
| 2 | **Vũ Quang Anh** | 23001831 | [@Quincy546](https://github.com/Quincy546) | Làm web (Frontend/Backend) |
| 3 | **Vũ Khánh Nam** | 23001907 | [@23001907-kn](https://github.com/23001907-kn) | Viết báo cáo (Chương 1, 2), kiểm định AI sửa chính tả |
| 4 | **Trịnh Thị Thu Huyền** | 23001889 | [@TrinhHuyen05](https://github.com/TrinhHuyen05) | Làm Slide, viết phần đánh giá và kết luận |
| 5 | **Đặng Chí Kiên** | 23001896 | [@K68A4](https://github.com/K68A4) | Không |

---

## ⚠️ Chuẩn bị Dữ liệu (Quan trọng)

Trước khi chạy hệ thống, bạn cần tải các tài nguyên cần thiết từ Google Drive và đặt vào đúng cấu trúc thư mục:

1. **Dữ liệu hệ thống**: Tải các thư mục `data/`, `artifacts/`, `logs/` và đặt tại thư mục gốc của dự án (`viecomrec/`).
   - `data/`: Chứa dữ liệu huấn luyện và database.
   - `artifacts/`: Chứa các model đã huấn luyện (ALS, BPR, BERT).
   - `logs/`: Chứa log hệ thống.

[Link Google Drive project](https://drive.google.com/drive/folders/1O7zOjDmeI3UIuzDWgUqPdxolqvA22dCk?usp=sharing)


2. **Hình ảnh sản phẩm**: Tải thư mục hình ảnh và đặt vào đường dẫn `web/server/public/`.
   - Điều này đảm bảo Web App hiển thị đúng hình ảnh sản phẩm.

[Link Google Drive ảnh sản phẩm](https://drive.google.com/drive/folders/1iZDfws0YvNXEv9mwzIlGgxf2AzveVmqG?usp=sharing)
## 🚀 Chạy Recommendation API (Docker)

Bạn có thể chạy ngay lõi hệ thống gợi ý mà không cần cài đặt môi trường Python phức tạp.

### 1. Pull Docker Image
```bash
docker pull maihoang07082005/viecomrec:latest
```

### 2. Chạy Container
Lệnh sau sẽ khởi động API server tại cổng 8000 và mount các thư mục dữ liệu đã chuẩn bị ở trên:

```bash
# Chạy API (cần mount data & artifacts từ local)
docker run -d -p 8000:8000 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/artifacts:/app/artifacts \
  -v ${PWD}/logs:/app/logs \
  maihoang07082005/viecomrec:latest
```
*(Lưu ý: Trên Windows PowerShell sử dụng `${PWD}`, trên Linux/macOS sử dụng `$(pwd)` hoặc đường dẫn tuyệt đối)*

### 3. Kiểm tra
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

> 📖 Xem hướng dẫn chi tiết tại [DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md).

## 🌐 Chạy Web Application (E-commerce)

Giao diện web mô phỏng sàn thương mại điện tử, kết nối với Recommendation API.

### 1. Yêu cầu
- **Node.js** (v16 trở lên)
- **MongoDB**:
  - Cách 1: Cài đặt MongoDB local.
  - Cách 2: Dùng Docker (khuyên dùng):
    ```bash
    cd web
    docker compose up -d mongodb
    ```

### 2. Cài đặt & Chạy nhanh
Sử dụng script tự động để cài đặt dependencies và khởi tạo dữ liệu mẫu:

**Windows:**
```powershell
cd web
.\scripts\setup.ps1
```

**macOS/Linux:**
```bash
cd web
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. Truy cập
- **Web Client**: http://localhost:3000
- **Web Server**: http://localhost:5000
- **Tài khoản Admin mặc định**: `admin@gmail.com` / `123456`

> 💡 **Lưu ý**: Web App cần kết nối với Recommendation API đang chạy ở port 8000 (Docker) để hiển thị gợi ý.

