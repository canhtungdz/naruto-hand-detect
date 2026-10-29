import os
import glob
from ultralytics import YOLO

def test_inference():
    print("="*80)
    print(" BẮT ĐẦU CHẠY KIỂM THỬ MÔ HÌNH TRÊN 10 ẢNH MẪU MỖI CLASS")
    print("="*80)
    
    # 1. Định nghĩa các đường dẫn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "naruto hand sign.v3i.yolov8")
    val_img_dir = os.path.join(dataset_dir, "valid", "images")
    val_lbl_dir = os.path.join(dataset_dir, "valid", "labels")
    model_path = os.path.join(current_dir, "runs", "detect", "naruto_yolov8n", "weights", "best_100_epoch.pt")
    
    # Danh sách class tương ứng với ID
    class_names = ['bird', 'boar', 'dog', 'dragon', 'hare', 'horse', 'monkey', 'ox', 'ram', 'rat', 'snake', 'tiger']
    
    if not os.path.exists(model_path):
        print(f"[!] LỖI: Không tìm thấy file model tại {model_path}")
        return
        
    print(f"[*] Nạp mô hình: {model_path}")
    model = YOLO(model_path)
    
    # 2. Thu thập 10 ảnh cho mỗi class từ tập validation
    # key: class_id, value: list of (image_path, label_path)
    class_samples = {i: [] for i in range(12)}
    
    label_files = glob.glob(os.path.join(val_lbl_dir, "*.txt"))
    
    for lbl_path in label_files:
        filename = os.path.basename(lbl_path)
        img_name_no_ext = os.path.splitext(filename)[0]
        
        # Tìm file ảnh tương ứng (jpg, jpeg, png)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            temp_path = os.path.join(val_img_dir, img_name_no_ext + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if not img_path:
            continue
            
        # Đọc class_id từ file label
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if class_id in class_samples and len(class_samples[class_id]) < 10:
                        # Tránh trùng lặp ảnh trong cùng một class list
                        if img_path not in [item[0] for item in class_samples[class_id]]:
                            class_samples[class_id].append((img_path, lbl_path))
                            
    # 3. Chạy thử nghiệm nhận diện với các ngưỡng Confidence khác nhau: 0.50, 0.35, 0.25
    conf_thresholds = [0.50, 0.35, 0.25]
    
    print("\n[*] Tiến hành chạy Inference...")
    
    # Lưu trữ kết quả đánh giá cho từng class ở từng ngưỡng conf
    # structure: {class_id: {conf: {"total": 0, "correct": 0, "wrong_class_counts": {}}}}
    evaluation = {}
    for cid in range(12):
        evaluation[cid] = {}
        for conf in conf_thresholds:
            evaluation[cid][conf] = {"total": 0, "correct": 0, "not_detected": 0, "wrong_predictions": {}}

    for cid in range(12):
        samples = class_samples[cid]
        cname = class_names[cid]
        
        if len(samples) == 0:
            print(f"[!] Cảnh báo: Không tìm thấy ảnh mẫu nào cho class '{cname}' trong tập validate.")
            continue
            
        for img_path, lbl_path in samples:
            # Chạy model dự đoán trên ảnh này
            results = model(img_path, verbose=False)[0]
            
            for conf in conf_thresholds:
                evaluation[cid][conf]["total"] += 1
                
                # Lọc các bounding box thỏa mãn ngưỡng conf hiện tại
                detected_boxes = []
                for box in results.boxes:
                    c_score = float(box.conf[0])
                    if c_score >= conf:
                        detected_boxes.append({
                            "class_id": int(box.cls[0]),
                            "conf": c_score
                        })
                        
                if len(detected_boxes) == 0:
                    evaluation[cid][conf]["not_detected"] += 1
                else:
                    # Lấy box có confidence cao nhất
                    best_box = sorted(detected_boxes, key=lambda x: x['conf'], reverse=True)[0]
                    pred_cid = best_box['class_id']
                    
                    if pred_cid == cid:
                        evaluation[cid][conf]["correct"] += 1
                    else:
                        pred_name = class_names[pred_cid]
                        wrong_dict = evaluation[cid][conf]["wrong_predictions"]
                        wrong_dict[pred_name] = wrong_dict.get(pred_name, 0) + 1

    # 4. In báo cáo chi tiết
    print("\n" + "="*95)
    print(" BÁO CÁO PHÂN TÍCH NHẬN DIỆN CHI TIẾT THEO CÁC NGƯỠNG CONFIDENCE")
    print("="*95)
    
    for conf in conf_thresholds:
        print(f"\n⚡ NGƯỠNG TIN CẬY (CONFIDENCE THRESHOLD): {conf:.2f}")
        print(f"{'Class thủ ấn':<15} | {'Mẫu thử':<8} | {'Nhận diện đúng':<15} | {'Không nhận diện':<16} | {'Nhầm sang Class khác (Số lượng)'}")
        print("-"*95)
        for cid in range(12):
            cname = class_names[cid]
            data = evaluation[cid][conf]
            total = data["total"]
            correct = data["correct"]
            not_det = data["not_detected"]
            wrongs = data["wrong_predictions"]
            
            if total == 0:
                print(f"{cname:<15} | {'0/0':<8} | {'N/A':<15} | {'N/A':<16} | N/A")
                continue
                
            wrong_str = ", ".join([f"{k}({v})" for k, v in wrongs.items()]) if wrongs else "Không bị nhầm"
            correct_rate = (correct / total) * 100
            
            print(f"{cname:<15} | {total:<8} | {correct}/{total} ({correct_rate:.0f}%) | {not_det}/{total} | {wrong_str}")
        print("-"*95)

if __name__ == "__main__":
    test_inference()
