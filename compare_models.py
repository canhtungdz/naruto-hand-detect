import os
import time
from ultralytics import YOLO

def evaluate_model(model_path, data_yaml):
    print(f"\n[*] Đang nạp và đánh giá mô hình: {os.path.basename(model_path)}...")
    try:
        model = YOLO(model_path)
        
        # Thực hiện đánh giá (Validation) trên tập validate
        # Sử dụng split='val' và verbose=False để chạy êm
        metrics = model.val(data=data_yaml, split='val', verbose=False)
        
        # Trích xuất các chỉ số chất lượng
        map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
        precision = metrics.results_dict.get('metrics/precision(B)', 0.0)
        recall = metrics.results_dict.get('metrics/recall(B)', 0.0)
        
        # Tốc độ xử lý trung bình (mili-giây trên mỗi frame ảnh)
        speed_inference = metrics.speed.get('inference', 0.0)
        
        return {
            "name": os.path.basename(model_path),
            "size_mb": os.path.getsize(model_path) / (1024 * 1024),
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "speed": speed_inference
        }
    except Exception as e:
        print(f"[!] Lỗi khi đánh giá {os.path.basename(model_path)}: {str(e)}")
        return None

def main():
    print("="*80)
    print(" BẮT ĐẦU ĐÁNH GIÁ VÀ SO SÁNH CHẤT LƯỢNG 2 MODEL (50 EPOCH vs 100 EPOCH)")
    print("="*80)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(current_dir, "configs", "data.yaml")
    weights_dir = os.path.join(current_dir, "runs", "detect", "naruto_yolov8n", "weights")
    
    model_50 = os.path.join(weights_dir, "best_50_epoch.pt")
    model_100 = os.path.join(weights_dir, "best_100_epoch.pt")
    
    results = []
    
    # Đánh giá mô hình 50 Epoch
    if os.path.exists(model_50):
        res = evaluate_model(model_50, data_yaml)
        if res:
            results.append(res)
    else:
        print(f"[!] Không tìm thấy file model: {model_50}")
        
    # Đánh giá mô hình 100 Epoch
    if os.path.exists(model_100):
        res = evaluate_model(model_100, data_yaml)
        if res:
            results.append(res)
    else:
        print(f"[!] Không tìm thấy file model: {model_100}")
        
    if len(results) == 0:
        print("[!] Không có mô hình hợp lệ nào để so sánh.")
        return
        
    # In bảng so sánh trực quan
    print("\n" + "="*95)
    print(" BẢNG SO SÁNH CHẤT LƯỢNG MÔ HÌNH THỰC TẾ (TRÊN TẬP VALIDATE)")
    print("="*95)
    print(f"{'Tên mô hình':<22} | {'Dung lượng':<12} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10} | {'Tốc độ xử lý':<15}")
    print("-"*95)
    for res in results:
        print(f"{res['name']:<22} | {res['size_mb']:.2f} MB    | {res['precision']:.4f}    | {res['recall']:.4f}    | {res['map50']:.4f}    | {res['map50_95']:.4f}    | {res['speed']:.2f} ms/frame")
    print("="*95)
    print("[*] Ghi chú: mAP50 & mAP50-95 càng GẦN 1.0 càng tốt. Tốc độ xử lý (ms/frame) càng THẤP càng nhanh.")
    print("="*95)

if __name__ == "__main__":
    main()
