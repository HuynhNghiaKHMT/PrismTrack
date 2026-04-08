File newtrack: Bản chính thức đạt kết quả tốt nhất trên MOT17. Các file đầy đủ nhất. code sạch sẽ nhất được triển khai tại đây. Thực nghiệm các modules sau:

0. KF     (New/Old),
1. CMC    (True/False),
2. ReID   (True/False),
3. ASSO   (Global/Local),
4. ASSI   (Multi/Joint), 
5. DEL    (True/False),
6. TAI    (True/False),
7. AFLink (True/False),
8. GBI    (True/False),
9. DLO    (True/False),
10. Overlap (IoU, GIoU, CIoU, DIoU, WMIoU, HMIoU).

Run:
python Tracker/run_newtrack.py --dataset "MOT17" --mode "val" 