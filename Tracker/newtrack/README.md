File newtrack: Bản chính thức đạt kết quả tốt nhất trên MOT17. Các file đầy đủ nhất. code sạch sẽ nhất được triển khai tại đây. Thực nghiệm các modules sau:

0. KF       (New/Old),
1. CMC      (True/False),
2. ReID     (True/False),
3. IDCBoost (True/False),
5. AFLink   (True/False),
6. GBI      (True/False),
7. Overlap (IoU, GIoU, CIoU, DIoU, WMIoU, HMIoU).
8. Cost (Motion, Velocity, Confidence, Shape, Appearance),
8. Cost (Motion, Velocity, Confidence, Shape, Appearance).

Note:
- run_newtrack.py: Phiên bản baseline, có thử nghiệm với low sequences và apdate logic tính chi phí dựa trên khoảng thời gian low sequences.
- run_newtrack_v1.py: Chỉnh sửa logic tính toán chi phí dựa trên vận tốc gốc & chỉnh sửa các ghi kết quả khi track được comfirmed.

Run:
python Tracker/run_newtrack.py --dataset "MOT17" --mode "val" 