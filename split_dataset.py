import os
import random
import shutil

def split_dataset():
    print("="*60)
    print(" BẮT ĐẦU CHIA LẠI DATASET TRÊN MÁY CỤC BỘ")
    print("="*60)

    # 1. Định nghĩa các đường dẫn thư mục
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "naruto hand sign.v3i.yolov8")
    
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
    
    valid_img_dir = os.path.join(dataset_dir, "valid", "images")
    valid_lbl_dir = os.path.join(dataset_dir, "valid", "labels")

    # Kiểm tra sự tồn tại của thư mục gốc train
    if not os.path.exists(train_img_dir) or not os.path.exists(train_lbl_dir):
        print("[!] LỖI: Không tìm thấy thư mục dữ liệu train tại:")
        print(f"    Images: {train_img_dir}")
        print(f"    Labels: {train_lbl_dir}")
        return

    # Đảm bảo các thư mục validate tồn tại
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_lbl_dir, exist_ok=True)

    # 2. Đếm số lượng file trước khi thực hiện
    train_images_before = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_images_before = [f for f in os.listdir(valid_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"[*] Số lượng ảnh ban đầu:")
    print(f"    - Tập Train: {len(train_images_before)} ảnh")
    print(f"    - Tập Validate: {len(valid_images_before)} ảnh")

    # 3. Tính toán số lượng ảnh cần chuyển (15% từ tập train)
    split_ratio = 0.15
    num_to_move = int(len(train_images_before) * split_ratio)
    
    if num_to_move == 0:
        print("[!] Không đủ ảnh trong tập train để thực hiện chia tách!")
        return

    # Lựa chọn ngẫu nhiên các ảnh cần di chuyển
    images_to_move = random.sample(train_images_before, num_to_move)

    print(f"[*] Tiến hành di chuyển ngẫu nhiên {num_to_move} ảnh sang tập Validate...")

    moved_count = 0
    for img_name in images_to_move:
        # File ảnh nguồn và đích
        src_img = os.path.join(train_img_dir, img_name)
        dst_img = os.path.join(valid_img_dir, img_name)
        
        # File nhãn tương ứng (.txt)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        src_lbl = os.path.join(train_lbl_dir, lbl_name)
        dst_lbl = os.path.join(valid_lbl_dir, lbl_name)

        try:
            # Di chuyển ảnh
            shutil.move(src_img, dst_img)
            
            # Di chuyển nhãn (nếu có)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            
            moved_count += 1
        except Exception as e:
            print(f"[!] Lỗi khi di chuyển file {img_name}: {str(e)}")

    # 4. Thống kê lại sau khi chia tách
    train_images_after = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_images_after = [f for f in os.listdir(valid_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print("\n" + "="*50)
    print(" HOÀN THÀNH CHIA LẠI DATASET CỤC BỘ!")
    print("="*50)
    print(f"[+] Đã di chuyển thành công: {moved_count} cặp file (Ảnh + Nhãn)")
    print(f"[*] Số lượng ảnh hiện tại:")
    print(f"    - Tập Train: {len(train_images_after)} ảnh (Giảm {len(train_images_before) - len(train_images_after)})")
    print(f"    - Tập Validate: {len(valid_images_after)} ảnh (Tăng {len(valid_images_after) - len(valid_images_before)})")
    print("="*60)

if __name__ == "__main__":
    split_dataset()
