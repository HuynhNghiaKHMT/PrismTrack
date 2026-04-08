File newtrack_tcm: Bản thử nghiệm với module TCM mới. Khi đưa score vào vector trạng thái của Kalman Filter, có thể cải thiện khả năng dự đoán và liên kết trong các tình huống khó khăn, đặc biệt khi đối tượng bị che khuất hoặc có nhiều nhiễu. Module này sẽ sử dụng score của detections để điều chỉnh quá trình cập nhật trạng thái của track, giúp tăng độ chính xác của việc liên kết giữa track và detection.

1. kalman_filter_hybird_new: vector của KF: [x, y, w, h, score vx, vy, vw, vh, vscore]
2. kalman_filter_hybird_score: vector trạng thái của KF gồm [score, score]
3. kalman_filter_hybird_original: vector trạng thái của KF gồm  [x, y, w, h, vx, vy, vw, vh]

Run:
python Tracker/run_newtrack_tcm.py --dataset "MOT17" --mode "val" 