import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

def advanced_cmc_visualizer(img_dir, pickle_path, delta_t=1, use_detection_masking=True, output_dir= "CMC"):
    """
    Hàm trực quan hóa từng bước thuật toán CMC động theo chuỗi thời gian (t -> t + delta_t).
    
    Tham số:
    - img_dir: Đường dẫn thư mục chứa ảnh (img1)
    - pickle_path: Đường dẫn tệp .pickle chứa bounding boxes
    - delta_t: Khoảng cách khung hình giữa ảnh gốc và ảnh đích (mặc định = 1)
    - use_detection_masking: True/False - Chọn vẽ ảnh che khuất đối tượng động hay không
    """
    output_dir= r"assets/" + output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Đọc dữ liệu phát hiện (detections)
    with open(pickle_path, 'rb') as f:
        detections_dict = pickle.load(f)
        
    img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    if len(img_names) <= delta_t:
        print(f"Lỗi: Số lượng ảnh trong thư mục không đủ với cấu hình delta_t = {delta_t}")
        return
        
    # Xác định Frame t và Frame t + delta_t
    img_name_t = img_names[0]
    img_name_tdelta = img_names[0 + delta_t]
    
    print(f"--- Đang thực thi CMC với Delta T = {delta_t} ---")
    print(f"Frame t: {img_name_t} | Frame t+dt: {img_name_tdelta}")
    
    # Đọc dữ liệu ảnh gốc
    frame_t = cv2.imread(os.path.join(img_dir, img_name_t))
    frame_tdelta = cv2.imread(os.path.join(img_dir, img_name_tdelta))
    
    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    gray_tdelta = cv2.cvtColor(frame_tdelta, cv2.COLOR_BGR2GRAY)
    h, w = gray_t.shape

    # Tham số cấu hình trích xuất đặc trưng góc Shi-Tomasi
    feature_params = dict(maxCorners=600, qualityLevel=0.01, minDistance=5, blockSize=3)

    # =========================================================================
    # Bước 1 & 1.5: Xử lý Frame t & Tùy chọn Mặt nạ Bounding Box
    # =========================================================================
    mask_t = np.zeros_like(gray_t)
    mask_t[int(0.02 * h): int(0.98 * h), int(0.02 * w): int(0.98 * w)] = 255
    
    # Lấy danh sách bbox của Frame t từ file pickle
    dets_t = detections_dict.get(img_name_t, [])
    if len(dets_t) == 0: dets_t = detections_dict.get(1, []) # Dự phòng nếu key lưu dạng số khung hình
    dets_t = np.array(dets_t)

    if use_detection_masking and len(dets_t) > 0:
        # TRỰC QUAN TRƯỚC KHI MASK: Trích xuất đặc trưng trên toàn ảnh (gồm cả người)
        kp_t_no_mask = cv2.goodFeaturesToTrack(gray_t, mask=None, **feature_params)
        img_before_mask = frame_t.copy()
        if kp_t_no_mask is not None:
            for pt in kp_t_no_mask:
                cv2.circle(img_before_mask, (int(pt[0][0]), int(pt[0][1])), 4, (0, 0, 255), -1) # Chấm đỏ cảnh báo dính nhiễu người
        cv2.imwrite(os.path.join(output_dir, "step1_5_before_masking.jpg"), img_before_mask)
        
        # Áp dụng mặt nạ: Xóa các vùng chứa người (gán bằng 0)
        frame_masked_visual = frame_t.copy()
        for det in dets_t:
            x1, y1, x2, y2 = map(int, det[:4])
            mask_t[y1:y2, x1:x2] = 0
            cv2.rectangle(frame_masked_visual, (x1, y1), (x2, y2), (0, 165, 255), 2) # Vẽ khung cam
            
        # TRỰC QUAN SAU KHI MASK: Điểm đặc trưng chỉ nằm ở nền tĩnh
        kp_t_with_mask = cv2.goodFeaturesToTrack(gray_t, mask=mask_t, **feature_params)
        img_after_mask = frame_masked_visual.copy()
        if kp_t_with_mask is not None:
            for pt in kp_t_with_mask:
                cv2.circle(img_after_mask, (int(pt[0][0]), int(pt[0][1])), 4, (0, 255, 0), -1) # Chấm xanh lá chuẩn nền
        cv2.imwrite(os.path.join(output_dir, "step1_5_after_masking.jpg"), img_after_mask)
        print("[Xong 1.5] Đã xuất ảnh so sánh Trước/Sau khi sử dụng Detection Masking.")
    else:
        # Nếu False, chỉ tính toán mặt nạ ngầm để trích xuất điểm nền cho ảnh 1
        for det in dets_t:
            if len(dets_t) > 0:
                x1, y1, x2, y2 = map(int, det[:4])
                mask_t[y1:y2, x1:x2] = 0

    # Ảnh 1: Điểm đặc trưng nền của Frame t
    kp_t = cv2.goodFeaturesToTrack(gray_t, mask=mask_t, **feature_params)
    img_step1 = frame_t.copy()
    if kp_t is not None:
        for pt in kp_t:
            cv2.circle(img_step1, (int(pt[0][0]), int(pt[0][1])), 4, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(output_dir, "step1_frame_t_keypoints.jpg"), img_step1)
    print("[Xong 1] Đã xuất ảnh trích xuất đặc trưng nền tại khung hình t.")

    # =========================================================================
    # Bước 2: Khung hình t+dt & Trực quan dịch chuyển bằng Optical Flow
    # =========================================================================
    # Tính luồng quang học dịch chuyển từ tọa độ gốc (kp_t) sang khung hình mới (gray_tdelta)
    kp_tdelta_matched, status, err = cv2.calcOpticalFlowPyrLK(gray_t, gray_tdelta, kp_t, None)
    
    img_step2 = frame_tdelta.copy()
    good_t = kp_t[status == 1]
    good_tdelta = kp_tdelta_matched[status == 1]
    
    # Vẽ tọa độ quá khứ (t) và hiện tại (t+dt) chồng lên nhau để thấy sự dịch chuyển nền
    for pt_t, pt_tdelta in zip(good_t, good_tdelta):
        p1 = (int(pt_t[0]), int(pt_t[1]))       # Vị trí cũ ở frame t
        p2 = (int(pt_tdelta[0]), int(pt_tdelta[1])) # Vị trí mới ở frame t+dt
        
        cv2.line(img_step2, p1, p2, (255, 255, 0), 1, cv2.LINE_AA) # Đường nối màu xanh cyan
        cv2.circle(img_step2, p1, 2, (0, 0, 255), -1)   # Vị trí cũ chấm đỏ
        cv2.circle(img_step2, p2, 3, (0, 255, 0), -1)   # Vị trí mới chấm xanh lá
        
    cv2.imwrite(os.path.join(output_dir, "step2_optical_flow_trajectory.jpg"), img_step2)
    print("[Xong 2] Đã xuất ảnh mô tả quỹ đạo dịch chuyển của điểm nền bằng Optical Flow.")

    # =========================================================================
    # Bước 3: Sử dụng RANSAC phân lọc Inliers (Xanh lá) và Outliers (Đỏ)
    # =========================================================================
    # RANSAC tìm kiếm mô hình chuyển động nền đồng nhất
    H, inliers_mask = cv2.estimateAffinePartial2D(good_t, good_tdelta, cv2.RANSAC, ransacReprojThreshold=3.0)
    
    img_step3 = frame_tdelta.copy()
    for i, (pt_t, pt_tdelta) in enumerate(zip(good_t, good_tdelta)):
        p2 = (int(pt_tdelta[0]), int(pt_tdelta[1]))
        p1 = (int(pt_t[0]), int(pt_t[1]))
        
        if inliers_mask[i][0] == 1:
            # Điểm thuộc về nền tĩnh khớp với chuyển động camera (Inlier)
            cv2.circle(img_step3, p2, 4, (0, 255, 0), -1) # Xanh lá
        else:
            # Điểm nhiễu động học hoặc chuyển động sai lệch (Outlier)
            cv2.circle(img_step3, p2, 4, (0, 0, 255), -1) # Vẽ màu Đỏ
            cv2.line(img_step3, p1, p2, (0, 0, 255), 1)
            
    cv2.imwrite(os.path.join(output_dir, "step3_ransac_filtering.jpg"), img_step3)
    print("[Xong 3] Đã xuất ảnh lọc nhiễu RANSAC phân định rõ rệt Inlier/Outlier.")

    # =========================================================================
    # Bước 4: In ma trận Affine, Hướng và Độ dịch pixel lên Frame t+dt
    # =========================================================================
    img_step4 = frame_tdelta.copy()
    
    # Trích xuất dữ liệu dịch chuyển từ ma trận Affine 2x3
    tx = H[0, 2] # Độ tịnh tiến theo phương ngang X (pixels)
    ty = H[1, 2] # Độ tịnh tiến theo phương đứng Y (pixels)
    
    # Tính toán góc hướng di chuyển tổng thể của camera (Tính theo Radian chuyển sang Độ)
    angle_rad = np.arctan2(ty, tx)
    angle_deg = np.degrees(angle_rad)
    displacement_magnitude = np.sqrt(tx**2 + ty**2) # Độ lớn vector tịnh tiến
    
    # Chuẩn bị văn bản hiển thị lên slide
    text_lines = [
        f"--- CMC Summary (Delta T = {delta_t}) ---",
        f"Affine Matrix H (2x3):",
        f"  [{H[0,0]:.4f}, {H[0,1]:.4f}, {H[0,2]:.2f}]",
        f"  [{H[1,0]:.4f}, {H[1,1]:.4f}, {H[1,2]:.2f}]",
        f"Horizontal Shift (tx): {tx:.2f} px",
        f"Vertical Shift (ty): {ty:.2f} px",
        f"Motion Magnitude: {displacement_magnitude:.2f} px",
        f"Camera Angle: {angle_deg:.1f} degrees"
    ]
    
    # In văn bản lên ảnh
    y_start = 40
    for line in text_lines:
        cv2.putText(img_step4, line, (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y_start += 28
        
    # Vẽ mũi tên hướng dịch chuyển lớn tại trung tâm để trực quan hóa
    center_pt = (w // 2, h // 2)
    end_pt = (int(w // 2 + tx * 3), int(h // 2 + ty * 3)) # Nhân hệ số 3 để dễ nhìn vector
    cv2.arrowedLine(img_step4, center_pt, end_pt, (0, 0, 255), 4, tipLength=0.2)
    
    cv2.imwrite(os.path.join(output_dir, "step4_final_metrics.jpg"), img_step4)
    print("[Xong 4] Đã xuất ảnh chứa thông tin Ma trận Affine và các thông số dịch chuyển.")
    print(f"--> Mọi bức ảnh đã được tổ chức tại thư mục: '{output_dir}'")

# --- ĐOẠN MẪU KHỞI CHẠY KIỂM THỬ ---
if __name__ == "__main__":
    IMAGE_DIR = r"assets/CMC"
    PICKLE_PATH = r"outputs/1. det/mot17_test_0.70.pickle"
    
    # Thử nghiệm với bước nhảy thời gian dài (Delta T = 5 khung hình) và bật trực quan mặt nạ đối tượng
    advanced_cmc_visualizer(img_dir=IMAGE_DIR, pickle_path=PICKLE_PATH, delta_t=1, use_detection_masking=True)