import os
import sys
from ultralytics import YOLO

def train_model():
    print("="*60)
    print(" KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN YOLOv8 - NARUTO HAND SIGN DETECT")
    print("="*60)
    
    # 1. Xác định các đường dẫn thư mục
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_yaml_path = os.path.join(project_root, "configs", "data.yaml")
    
    print(f"[*] Project root directory: {project_root}")
    print(f"[*] Dataset config path: {data_yaml_path}")
    
    if not os.path.exists(data_yaml_path):
        print(f"[!] LỖI: Không tìm thấy file cấu hình dữ liệu tại {data_yaml_path}")
        sys.exit(1)
        
    # 2. Khởi tạo mô hình YOLOv8 Nano pre-trained
    print("[*] Đang tải mô hình YOLOv8 Nano pre-trained...")
    model = YOLO("yolov8n.pt")
    
    # 3. Cấu hình huấn luyện
    # Tự động phát hiện nếu chạy trên môi trường Google Colab có GPU
    is_colab = "COLAB_GPU" in os.environ or os.path.exists("/content")
    device = "0" if is_colab else "cpu"
    epochs = 100 if is_colab else 1  # Máy local chỉ chạy thử 1 epoch để xác minh liên kết dữ liệu
    
    print(f"[*] Môi trường phát hiện: {'Google Colab (GPU)' if is_colab else 'Local (CPU)'}")
    print(f"[*] Thiết bị huấn luyện sẽ sử dụng: {device.upper()}")
    print(f"[*] Số lượng Epoch thiết lập: {epochs}")
    print("[*] Bắt đầu huấn luyện...")
    
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device=device,
            workers=2,
            name="naruto_yolov8n",
            project=os.path.join(project_root, "runs") # Lưu kết quả vào thư mục runs ở gốc dự án
        )
        print("="*60)
        print(" HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
        print(f"[*] Kết quả huấn luyện được lưu tại thư mục: runs/train/naruto_yolov8n")
        print(f"[*] File trọng số tốt nhất (best.pt) sẽ nằm trong thư mục weights/")
        print("="*60)
    except Exception as e:
        print(f"[!] Đã xảy ra lỗi trong quá trình huấn luyện: {str(e)}")
        print("[!] Vui lòng kiểm tra lại cấu trúc dataset và file configs/data.yaml")

if __name__ == "__main__":
    train_model()
