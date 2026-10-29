import cv2
import numpy as np

def draw_semi_transparent_rect(frame, pt1, pt2, color, alpha=0.6):
    """Vẽ hình chữ nhật bán trong suốt (semi-transparent overlay)"""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_detections(frame, detections):
    """
    Vẽ Bounding Box của các thủ ấn phát hiện được.
    Sử dụng màu sắc neon nổi bật (cyan).
    """
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['class'].upper()
        conf = det['conf']
        
        # Thiết lập màu sắc neon sáng (Cyan: (255, 255, 0) trong OpenCV là BGR)
        color = (255, 255, 0) 
        
        # 1. Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 2. Vẽ nhãn & confidence bán trong suốt phía trên box
        label_text = f"{label} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Đảm bảo box label không bị tràn khỏi khung hình phía trên
        label_y1 = max(y1 - h - 10, 0)
        label_y2 = y1
        
        draw_semi_transparent_rect(frame, (x1, label_y1), (x1 + w + 10, label_y2), (40, 40, 40), alpha=0.7)
        cv2.putText(frame, label_text, (x1 + 5, label_y2 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_status_bar(frame, current_chain, tracking_sign, frame_counter, min_required):
    """
    Vẽ dải trạng thái chuỗi ấn tay hiện tại ở góc trên màn hình.
    Thiết kế tinh xảo, nền bán trong suốt sang trọng.
    """
    h, w, _ = frame.shape
    
    # 1. Vẽ dải banner bán trong suốt phía trên
    draw_semi_transparent_rect(frame, (0, 0), (w, 50), (20, 20, 20), alpha=0.7)
    
    # 2. Vẽ dòng chữ "CHUỖI ẤN TAY"
    cv2.putText(frame, "CHUOI AN TAY: ", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # 3. Vẽ các thủ ấn đã tích lũy
    if len(current_chain) > 0:
        chain_str = " -> ".join([s.upper() for s in current_chain])
        cv2.putText(frame, chain_str, (160, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Trong...", (160, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
                    
    # 4. Hiển thị tiến trình lọc nhiễu của ấn hiện tại (Progress indicator) ở góc trên bên phải
    if tracking_sign:
        progress = int((frame_counter / min_required) * 100)
        progress = min(progress, 100)
        progress_text = f"Dang giu: {tracking_sign.upper()} ({progress}%)"
        cv2.putText(frame, progress_text, (w - 240, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

def draw_jutsu_activation(frame, active_jutsu):
    """
    Hiển thị hiệu ứng Nhẫn thuật đặc biệt khi kích hoạt thành công.
    Vẽ một tấm overlay trung tâm rực rỡ với hiệu ứng ánh sáng.
    """
    if not active_jutsu:
        return
        
    h, w, _ = frame.shape
    
    # 1. Vẽ viền nhấp nháy màu đỏ/cam xung quanh toàn bộ khung hình camera (Hiệu ứng vận Chakra!)
    border_color = (0, 140, 255) if "Hoa" in active_jutsu['element'] else (255, 120, 0)
    cv2.rectangle(frame, (0, 0), (w, h), border_color, 8)
    
    # 2. Vẽ bảng thông báo bán trong suốt ở trung tâm màn hình
    center_x, center_y = w // 2, h // 2
    box_w, box_h = 420, 120
    pt1 = (center_x - box_w // 2, center_y - box_h // 2)
    pt2 = (center_x + box_w // 2, center_y + box_h // 2)
    
    draw_semi_transparent_rect(frame, pt1, pt2, (15, 15, 15), alpha=0.8)
    # Vẽ viền cho bảng thông báo
    cv2.rectangle(frame, pt1, pt2, border_color, 2)
    
    # 3. Vẽ dòng tiêu đề Nhẫn thuật (Hệ nguyên tố)
    el_text = f"=== {active_jutsu['element'].upper()} ==="
    (el_w, el_h), _ = cv2.getTextSize(el_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, el_text, (center_x - el_w // 2, center_y - box_h // 2 + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)
                
    # 4. Vẽ tên Nhẫn thuật lớn và rực rỡ ở giữa bảng
    name_text = active_jutsu['name_vi'].upper()
    (name_w, name_h), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(frame, name_text, (center_x - name_w // 2, center_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
