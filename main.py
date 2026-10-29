import cv2
import time
import os
import sys

from src.detector import HandSignDetector
from src.jutsu_manager import JutsuManager
from src.gui import draw_detections, draw_status_bar, draw_jutsu_activation

def main():
    print("="*60)
    print("  ỨNG DỤNG NHẬN DIỆN THỦ ẤN VÀ THI TRIỂN NHẪN THUẬT NARUTO  ")
    print("="*60)
    
    # 1. Khởi tạo bộ nhận diện và quản lý nhẫn thuật
    try:
        detector = HandSignDetector()
        manager = JutsuManager()
    except Exception as e:
        print(f"[!] Lỗi khởi tạo ứng dụng: {str(e)}")
        sys.exit(1)
        
    # 2. Mở kết nối Webcam
    print("[*] Đang khởi động Webcam...")
    cap = cv2.VideoCapture(0) # 0 là webcam mặc định
    
    # Đặt độ phân giải hợp lý để cân bằng tốc độ CPU (640x480 là tối ưu nhất cho YOLOv8)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[!] LỖI: Không thể kết nối với Webcam. Vui lòng kiểm tra quyền camera!")
        sys.exit(1)
        
    print("[+] Webcam đã hoạt động. Đưa tay trước camera để làm thủ ấn!")
    print("[*] Nhấn phím 'Q' trên bàn phím để tắt ứng dụng.")
    
    # Các biến tính toán FPS
    prev_time = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Không thể đọc khung hình từ Webcam. Đang thử lại...")
            continue
            
        # Lật ngược khung hình theo chiều ngang (Mirror effect) giúp người dùng dễ tương tác cử chỉ tay
        frame = cv2.flip(frame, 1)
        
        # 3. Chạy nhận diện thủ ấn (Tăng ngưỡng tin cậy confidence lên lại 0.5 để giảm thiểu nhận diện nhầm giữa horse và dragon)
        detections = detector.detect(frame, conf_threshold=0.5)
        
        # 4. Trích xuất thủ ấn có độ tin cậy cao nhất để đưa vào bộ lọc logic nhẫn thuật
        best_match_sign = None
        if len(detections) > 0:
            # detections đã được sắp xếp từ cao xuống thấp trong detector.py
            best_match_sign = detections[0]['class']
            
        # 5. Cập nhật thủ ấn vào bộ quản lý nhẫn thuật
        active_jutsu = manager.update(best_match_sign)
        
        # 6. Vẽ giao diện và overlay thông tin lên khung hình
        # Vẽ các bounding boxes
        draw_detections(frame, detections)
        
        # Vẽ dải thông tin chuỗi ấn tích lũy phía trên cùng
        draw_status_bar(frame, manager.current_chain, manager.tracking_sign, 
                        manager.frame_counter, manager.MIN_REQUIRED_FRAMES)
                        
        # Vẽ hiệu ứng kích hoạt nhẫn thuật nếu có
        if active_jutsu:
            draw_jutsu_activation(frame, active_jutsu)
            
        # 7. Tính toán và vẽ chỉ số FPS thời gian thực lên góc dưới màn hình
        current_time = time.time()
        time_diff = current_time - prev_time
        prev_time = current_time
        if time_diff > 0:
            fps = 1 / time_diff
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
        # 8. Hiển thị khung hình ra cửa sổ
        cv2.imshow("Naruto Hand Sign Jutsu Detection", frame)
        
        # Bắt phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
            
    # Giải phóng webcam và dọn dẹp tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Đã tắt Webcam và đóng ứng dụng an toàn. Hẹn gặp lại bạn!")

if __name__ == "__main__":
    main()
