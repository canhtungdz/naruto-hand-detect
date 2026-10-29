import os
import yaml
import time

class JutsuManager:
    def __init__(self, jutsu_config_path=None):
        """
        Khởi tạo bộ quản lý nhẫn thuật và thuật toán lọc nhiễu thời gian thực.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        
        # Đường dẫn mặc định đến file cấu hình nhẫn thuật
        default_config_path = os.path.join(project_root, "configs", "jutsu.yaml")
        self.config_path = jutsu_config_path if jutsu_config_path else default_config_path
        
        self.jutsus = self._load_jutsus()
        
        # Trạng thái tích lũy chuỗi
        self.current_chain = []          # Chuỗi các thủ ấn hợp lệ đã ghi nhận
        self.last_action_time = time.time() # Thời gian ghi nhận ấn cuối cùng
        
        # Bộ lọc Temporal Filter & Debounce với cơ chế dung sai
        self.tracking_sign = None        # Thủ ấn đang theo dõi liên tục ở các frame gần đây
        self.frame_counter = 0           # Bộ đếm số frame liên tiếp nhận diện cùng 1 thủ ấn
        self.last_recorded_sign = None   # Thủ ấn hợp lệ vừa được thêm vào chuỗi gần nhất
        
        self.missed_frame_counter = 0    # Đếm số frame trống liên tiếp
        self.MAX_MISSED_FRAMES = 4       # Cho phép tối đa 4 frame trống liên tiếp mà không reset bộ lọc (dung sai camera)
        
        self.new_sign_candidate = None   # Nhãn ứng viên mới đề xuất chuyển đổi
        self.new_sign_counter = 0        # Đếm số frame liên tiếp xuất hiện nhãn mới
        self.NEW_SIGN_THRESHOLD = 2      # Phải có ít nhất 2 frame liên tiếp của nhãn mới để xác nhận đổi nhãn (tránh nhiễu 1 frame)
        
        # Cấu hình ngưỡng lọc nhiễu
        self.MIN_REQUIRED_FRAMES = 5     # Số frame liên tiếp tối thiểu để công nhận thủ ấn
        self.TIMEOUT_LIMIT = 3.0         # Thời gian tối đa (giây) chờ giữa các ấn trước khi reset chuỗi
        
        # Trạng thái nhẫn thuật đang kích hoạt (để vẽ hiệu ứng kéo dài 2 giây)
        self.active_jutsu = None
        self.active_jutsu_duration = 2.0 # Thời gian hiển thị hiệu ứng nhẫn thuật (giây)

    def _load_jutsus(self):
        """Tải danh sách nhẫn thuật từ file cấu hình yaml."""
        if not os.path.exists(self.config_path):
            print(f"[!] CẢNH BÁO: Không tìm thấy file cấu hình nhẫn thuật tại {self.config_path}")
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                print(f"[+] Đã tải danh sách {len(data) if data else 0} nhẫn thuật thành công!")
                return data if data else {}
        except Exception as e:
            print(f"[!] LỖI: Không thể đọc cấu hình nhẫn thuật. Chi tiết: {str(e)}")
            return {}

    def update(self, detected_sign):
        """
        Cập nhật trạng thái thủ ấn phát hiện được ở frame hiện tại vào thuật toán lọc nhiễu.
        detected_sign: Nhãn thủ ấn nhận diện được (ví dụ: 'tiger', hoặc None nếu không nhận diện được gì)
        """
        current_time = time.time()
        
        # 1. Kiểm tra trạng thái Nhẫn thuật đang kích hoạt (xóa sau khi hết hạn 2 giây)
        if self.active_jutsu and (current_time - self.active_jutsu['timestamp'] > self.active_jutsu_duration):
            self.active_jutsu = None

        # 2. Cơ chế Reset theo thời gian chờ (Timeout Reset)
        if len(self.current_chain) > 0:
            if current_time - self.last_action_time > self.TIMEOUT_LIMIT:
                print("[*] Quá thời gian chờ (3s) ➔ Đã tự động reset chuỗi thủ ấn.")
                self.reset_chain()

        # 3. Thuật toán Lọc nhiễu thời gian thực (Temporal Filtering) với dung sai hụt khung hình và làm mịn nhãn
        if detected_sign is not None:
            self.missed_frame_counter = 0 # Có nhận diện ➔ Reset bộ đếm frame trống
            
            if self.tracking_sign is None:
                # Bắt đầu theo dõi thủ ấn mới
                self.tracking_sign = detected_sign
                self.frame_counter = 1
                self.new_sign_candidate = None
                self.new_sign_counter = 0
            elif detected_sign == self.tracking_sign:
                self.frame_counter += 1
                self.new_sign_candidate = None
                self.new_sign_counter = 0
                
                # Nếu phát hiện đủ số khung hình liên tiếp để công nhận
                if self.frame_counter >= self.MIN_REQUIRED_FRAMES:
                    # Chặn trùng lặp: Không ghi nhận liên tiếp cùng 1 thủ ấn vào chuỗi
                    if detected_sign != self.last_recorded_sign:
                        self.current_chain.append(detected_sign)
                        self.last_recorded_sign = detected_sign
                        self.last_action_time = current_time
                        print(f"[+] Đã ghi nhận thủ ấn hợp lệ: {detected_sign.upper()} ➔ Chuỗi hiện tại: {self.current_chain}")
                        
                        # So khớp chuỗi vừa cập nhật xem có kích hoạt nhẫn thuật nào không
                        self._check_jutsu_match(current_time)
                        
                    # Reset bộ đếm frame của ấn hiện tại sau khi đã xử lý
                    self.frame_counter = 0
            else:
                # Nhận diện khác với tracking_sign hiện tại ➔ Lọc nhiễu chuyển đổi nhãn (Label Smoothing)
                # Thay vì đổi nhãn ngay lập tức, ta cần nó xuất hiện liên tiếp vài frame để thực sự xác nhận đổi
                if detected_sign == self.new_sign_candidate:
                    self.new_sign_counter += 1
                else:
                    self.new_sign_candidate = detected_sign
                    self.new_sign_counter = 1
                    
                if self.new_sign_counter >= self.NEW_SIGN_THRESHOLD:
                    # Đã xác nhận thực sự đổi sang ấn tay mới
                    self.tracking_sign = detected_sign
                    self.frame_counter = self.new_sign_counter
                    self.new_sign_candidate = None
                    self.new_sign_counter = 0
        else:
            # Không phát hiện gì ở frame này (detected_sign is None)
            # Áp dụng cơ chế dung sai hụt khung hình (Missed Frame Tolerance)
            if self.tracking_sign is not None:
                self.missed_frame_counter += 1
                if self.missed_frame_counter >= self.MAX_MISSED_FRAMES:
                    # Thực sự mất dấu tay ➔ Tiến hành reset bộ đếm lọc nhiễu
                    self.frame_counter = 0
                    self.tracking_sign = None
                    self.missed_frame_counter = 0
                    self.new_sign_candidate = None
                    self.new_sign_counter = 0
            else:
                self.frame_counter = 0
                self.missed_frame_counter = 0

        return self.active_jutsu

    def _check_jutsu_match(self, current_time):
        """So khớp chuỗi thủ ấn hiện tại với các chuỗi nhẫn thuật trong cấu hình."""
        for key, jutsu_data in self.jutsus.items():
            sequence = jutsu_data.get('sequence', [])
            
            # Kiểm tra xem chuỗi ấn tay hiện tại có khớp hoàn toàn với nhẫn thuật không
            if self.current_chain == sequence:
                print(f"[💥] KÍCH HOẠT THÀNH CÔNG: {jutsu_data.get('name_vi', key)}!")
                self.active_jutsu = {
                    "key": key,
                    "name_vi": jutsu_data.get('name_vi', key),
                    "element": jutsu_data.get('element', 'Vô hệ'),
                    "timestamp": current_time
                }
                # Tự động dọn dẹp trạng thái sau khi thi triển thành công
                self.reset_chain()
                break

    def reset_chain(self):
        """Xóa sạch chuỗi ấn hiện tại và reset các biến trạng thái lọc."""
        self.current_chain = []
        self.last_recorded_sign = None
        self.tracking_sign = None
        self.frame_counter = 0
        self.missed_frame_counter = 0
        self.new_sign_candidate = None
        self.new_sign_counter = 0

# --- UNIT TEST GIẢ LẬP ĐỂ XÁC MINH LOGIC (Tự động chạy khi gọi trực tiếp file này) ---
if __name__ == "__main__":
    print("="*60)
    print(" CHẠY KIỂM THỬ ĐỘC LẬP MODULE JUTSU_MANAGER")
    print("="*60)
    
    # Khởi tạo manager giả lập file cấu hình
    manager = JutsuManager()
    
    # 1. Giả lập đưa tay làm ấn 'ox' liên tiếp trong 6 frame (đủ 5 frame để ghi nhận)
    print("\n[*] Giả lập làm ấn 'ox' liên tiếp 6 frame...")
    for _ in range(6):
        manager.update('ox')
    
    # 2. Giả lập làm ấn 'hare' liên tiếp trong 6 frame
    print("\n[*] Giả lập làm ấn 'hare' liên tiếp 6 frame...")
    for _ in range(6):
        manager.update('hare')
        
    # 3. Giả lập đưa tay làm ấn 'monkey' liên tiếp trong 6 frame ➔ Kỳ vọng kích hoạt Chidori (ox -> hare -> monkey)
    print("\n[*] Giả lập làm ấn 'monkey' liên tiếp 6 frame...")
    active_jutsu = None
    for _ in range(6):
        active_jutsu = manager.update('monkey')
        
    if active_jutsu:
        print(f"\n[✔️] KẾT QUẢ TEST: Kích hoạt thành công nhẫn thuật: {active_jutsu['name_vi']}")
    else:
        print("\n[❌] KẾT QUẢ TEST THẤT BẠI: Không kích hoạt được nhẫn thuật.")
    print("="*60)
