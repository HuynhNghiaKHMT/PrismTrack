File tracktrack_exp: Thực nghiệm các module mới của TrackTrack trên MOT17 validation set. Các module bao gồm: 
0. KF     (New/Old),
1. CMC    (True/False),
2. ReID   (True/False),
3. ASSO   (Global/Local),
4. ASSI   (Multi/Joint), 
5. DEL    (True/False),
6. TAI    (True/False),
7. AFLink (True/False),
8. GBI    (True/False),

Run:
python Tracker/run_tracktrack_exp.py --dataset "MOT17" --mode "val" 