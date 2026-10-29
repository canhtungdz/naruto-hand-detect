import os
import cv2
from ultralytics import YOLO

class HandSignDetector:
    def __init__(self, model_path=None):
        """
        Khởi tạo bộ nhận diện thủ ấn Naruto sử dụng mô hình YOLOv8.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        
        # 1. Định nghĩa các đường dẫn file weights có thể có (ưu tiên best_100_epoch.pt)
        weights_dir = os.path.join(project_root, "runs", "detect", "naruto_yolov8n", "weights")
        
        possible_paths = [
            os.path.join(weights_dir, "best_50_epoch.pt"), # Ưu tiên số 1: Model 100 Epochs (trong folder weights)
            os.path.join(project_root, "best_100_epoch.pt"),# Ưu tiên số 2: Model 100 Epochs (thư mục gốc)
            os.path.join(weights_dir, "best_50_epoch.pt"),  # Ưu tiên số 3: Model 50 Epochs (trong folder weights)
            os.path.join(project_root, "best_50_epoch.pt"), # Ưu tiên số 4: Model 50 Epochs (thư mục gốc)
            os.path.join(weights_dir, "best.pt"),           # Ưu tiên số 5: Bản best.pt mặc định (trong folder weights)
            os.path.join(project_root, "best.pt")            # Ưu tiên số 6: Bản best.pt mặc định (thư mục gốc)
        ]
        
        selected_path = None
        for path in possible_paths:
            if os.path.exists(path):
                selected_path = path
                break
                
        if model_path is not None:
            self.model_path = model_path
        elif selected_path is not None:
            self.model_path = selected_path
            print(f"[*] Tìm thấy mô hình huấn luyện tại: {self.model_path}")
        else:
            # Fallback về model mặc định yolov8n.pt để chạy thử nghiệm nếu chưa huấn luyện
            self.model_path = "yolov8n.pt"
            print(f"[!] CẢNH BÁO: Chưa tìm thấy mô hình đã huấn luyện nào trong các đường dẫn tìm kiếm.")
            print(f"[*] Hệ thống tự động sử dụng mô hình gốc '{self.model_path}' để chạy thử nghiệm giao diện.")

        try:
            self.model = YOLO(self.model_path)
            print(f"[+] Đã tải mô hình YOLOv8 thành công từ: {self.model_path}")
        except Exception as e:
            print(f"[!] LỖI: Không thể tải mô hình YOLOv8. Chi tiết: {str(e)}")
            raise e

    def detect(self, frame, conf_threshold=0.5):
        """
        Nhận diện các thủ ấn trong khung hình.
        Trả về danh sách các bounding boxes và tên thủ ấn phát hiện được.
        """
        if frame is None:
            return []

        # Chạy inference
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
                
            # Lấy tọa độ bounding box (x1, y1, x2, y2)
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Lấy class ID và nhãn class tương ứng
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            detections.append({
                "box": [x1, y1, x2, y2],
                "class": class_name,
                "conf": conf
            })
            
        # Sắp xếp các nhận diện theo độ tin cậy giảm dần
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        return detections
