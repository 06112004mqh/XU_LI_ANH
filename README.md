# TÁI TẠO SƠ ĐỒ KẾT NỐI TỪ ẢNH CHỤP
PROJECT môn : Xử lí ảnh và thị giác Robot
# Công Cụ Phân Tích & Tái Tạo Sơ Đồ Nguyên Lý PCB (PCB Schematic Reconstruction)

Dự án này là một ứng dụng phần mềm sử dụng **Deep Learning (YOLOv8)** kết hợp với **Xử lý ảnh (Computer Vision)** để tự động phân tích hình ảnh bo mạch in (PCB). Công cụ giúp nhận diện linh kiện, tách đường mạch đồng và tái tạo lại sơ đồ kết nối (Netlist) dưới dạng đồ thị.

## 🚀 Tính Năng Nổi Bật

*   **Giao diện đồ họa (GUI)**: Hiện đại, hỗ trợ Dark Mode, được xây dựng bằng `CustomTkinter`.
*   **Xử lý tiền kỳ**:
    *   Tự động căn chỉnh góc nghiêng ảnh (Perspective Correction).
    *   Bộ lọc tùy chỉnh độ sáng, độ tương phản và khử nhiễu để làm rõ đường mạch.
*   **Nhận diện linh kiện**: Tích hợp **YOLOv8** và thư viện **SAHI** (Slicing Aided Hyper Inference) để nhận diện chính xác các linh kiện nhỏ trên ảnh độ phân giải cao.
*   **Trích xuất đường mạch (Trace Extraction)**: Thuật toán xử lý ảnh giúp tách nền, làm mảnh (skeletonize) đường mạch và tự động nối các đoạn mạch bị đứt gãy.
*   **Tái tạo đồ thị (Graph Reconstruction)**: Xây dựng đồ thị kết nối logic giữa các linh kiện (chân linh kiện nối với nhau qua đường mạch nào).
*   **Trực quan hóa**: Hiển thị kết quả chi tiết với Matplotlib (Box linh kiện, Mask đường mạch, Skeleton và Đồ thị kết nối).

## 📂 Cấu Trúc Thư Mục

Để chương trình hoạt động chính xác, vui lòng sắp xếp thư mục như sau:

```text
PCB-Reconstruction-Project/
│
├── models/
│   └── best.pt                  # File weights của model YOLOv8 (bạn cần có file này)
│
├── src/                         # Thư mục chứa các module xử lý logic
│   ├── __init__.py              # (Tùy chọn)
│   ├── detector.py              # Class nhận diện linh kiện (dùng SAHI + YOLO)
│   ├── trace_extractor_v2.py    # Class xử lý ảnh và tách đường mạch
│   ├── graph_builder_v2.py      # Class xây dựng đồ thị kết nối (NetworkX)
│   ├── config.py                # Các cấu hình tham số
│   └── utils.py                 # Các hàm hỗ trợ hiển thị
│
├── main_v3.py                   # File chạy chính của chương trình (GUI)
├── requirements.txt             # Danh sách thư viện cần cài đặt
└── README.md                    # File hướng dẫn này

🛠️ Cài Đặt
Clone dự án về máy:
code
Bash
git clone https://github.com/username-cua-ban/pcb-reconstruction.git
cd pcb-reconstruction
Cài đặt các thư viện cần thiết:
Khuyên dùng môi trường ảo (Virtual Environment) hoặc Conda.
code
Bash
pip install -r requirements.txt
(Nếu chưa có file requirements.txt, xem danh sách thư viện ở cuối file này)
Cấu hình Model:
Copy file weights (best.pt) bạn đã train vào thư mục models/.
Quan trọng: Mở file main_v3.py, tìm dòng khai báo self.model_path (khoảng dòng 45) và sửa lại đường dẫn cho đúng với máy của bạn (nên dùng đường dẫn tương đối):
code
Python
# Sửa lại thành:
self.model_path = os.path.join("models", "best.pt")
🖥️ Hướng Dẫn Sử Dụng
Chạy chương trình:
code
Bash
python main_v3.py
Các bước thao tác trên giao diện:
Bước 1 - Load Image: Nhấn nút 📂 Load Image để chọn ảnh PCB (nên chọn ảnh chụp thẳng góc từ trên xuống).
Bước 2 - Preprocessing (Xử lý ảnh):
Nhấn 📐 Auto Calib nếu ảnh bị nghiêng.
Điều chỉnh thanh trượt Brightness, Contrast, Denoise sao cho đường mạch hiện rõ nhất và tách biệt với nền.
Bước 3 - Analyze: Nhấn nút màu xanh ⚡ ANALYZE PCB. Quá trình này có thể mất vài giây tùy vào độ phân giải ảnh.
Bước 4 - Xem kết quả: Chương trình sẽ tự động chuyển sang tab "Analysis Result". Tại đây bạn có thể xem:
Ảnh phát hiện linh kiện.
Mask đường mạch.
Đường mạch dạng khung xương (Skeleton).
Đồ thị kết nối các linh kiện.
🧠 Nguyên Lý Hoạt Động
Detector (src/detector.py): Sử dụng thư viện SAHI để cắt ảnh lớn thành các mảnh nhỏ (slices), sau đó dùng model YOLOv8 để nhận diện linh kiện. Cách này giúp không bỏ sót linh kiện nhỏ.
Trace Extractor (src/trace_extractor_v2.py): Sử dụng thuật toán thích ứng (CLAHE, Adaptive Threshold) để tách màu đồng. Sau đó dùng thuật toán Skeletonize để thu nhỏ đường mạch về độ rộng 1 pixel.
Graph Builder (src/graph_builder_v2.py):
Tìm các vùng liên thông (Connected Components) trên đường mạch (gọi là Net).
Kiểm tra va chạm (Touching logic): Nếu một Net chạm vào Bounding Box của linh kiện nào, nó sẽ tạo liên kết (Edge) giữa linh kiện đó và Net trong đồ thị.
📋 Yêu Cầu Hệ Thống & Thư Viện
Python 3.8 trở lên
Các thư viện chính (copy vào requirements.txt):
code
Text
customtkinter
opencv-python
numpy
Pillow
matplotlib
networkx
ultralytics
sahi
torch
torchvision
⚠️ Lưu Ý
Chương trình hoạt động tốt nhất với PCB 1 lớp hoặc 2 lớp mà đường mạch lộ rõ ra bên ngoài.
Ảnh đầu vào cần có độ nét cao và ánh sáng đều. Bóng đổ quá nhiều có thể làm đứt đường mạch trong quá trình xử lý ảnh.
Cần sửa lại chính xác link model 
🤝 Đóng Góp
Mọi đóng góp, báo lỗi hoặc yêu cầu tính năng mới đều được hoan nghênh. Vui lòng tạo Issue hoặc Pull Request trên GitHub.
Do link model nhan dien linh kien khá lớn ko thể up lên github: https://drive.google.com/drive/folders/13cxrhgwhS2qnKPX2p5SUxkj_z7OKquZm?dmr=1&ec=wgc-drive-globalnav-goto
