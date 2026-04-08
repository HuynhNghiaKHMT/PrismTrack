File newtrack_oru: Bản thử nghiệm với module ORU mới. Các file đầy đủ nhất. code sạch sẽ nhất được triển khai tại đây. Thực nghiệm các modules sau:Ý tưởng sử dụng last observed direction để hỗ trợ việc liên kết track và detection, đặc biệt trong các tình huống có nhiều đối tượng tương tự nhau hoặc khi đối tượng bị che khuất một phần. Module này sẽ tính toán hướng di chuyển của track dựa trên các vị trí trước đó và sử dụng thông tin này để cải thiện quá trình liên kết.

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
10. ORU   (True/False),

Run:
python Tracker/run_newtrack_oru.py --dataset "MOT17" --mode "val" 