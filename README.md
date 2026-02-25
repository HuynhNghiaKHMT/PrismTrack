# SmartTrack

## Prepare
**1. Downlodad datasets**
  - MOT17: https://motchallenge.net/data/MOT17.zip
  - MOT20: https://motchallenge.net/data/MOT20.zip
  - DanceTrack: https://dancetrack.github.io/

<br />

**2. Locate codes and datasets as below**
```bash
Smart_Track
├── YOLOX/
    ├── exps
    ├── jsons
    ├── weights
    ├── yolox
    └── detect.py
├── FastReID/
    ├── configs
    ├── fastreid
    ├── weights
    ├── train_net.py
    └── ext_feats.py
├── Track/
    ├── AFLink/
    ├── Trackers/
    ├── trackeval/
    ├── utils/
    └── run.py/
├── dataset/
    ├── MOT17
    ├── MOT20
    └── DanceTrack
├── outputs/
    ├── 1. det
    ├── 2. det_feat
    └── 3. track
├── .gitignore
├── requirements.txt
└── README.md
```

<br />

**3. Run**
1. YOLOX
```
# For MOT17 validation
python YOLOX/detect.py --dataset 'MOT17' --mode 'val' --nms 0.80 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT17' --mode 'val' --nms 0.95 -b 1 -d 1 --fp16 --fuse

# For MOT17 test
python YOLOX/detect.py --dataset 'MOT17' --mode 'test' --nms 0.80 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT17' --mode 'test' --nms 0.95 -b 1 -d 1 --fp16 --fuse

# For MOT20 validation
python YOLOX/detect.py --dataset 'MOT20' --mode 'val' --nms 0.80 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT20' --mode 'val' --nms 0.95 -b 1 -d 1 --fp16 --fuse

# For MOT20 test
python YOLOX/detect.py --dataset 'MOT20' --mode 'test' --nms 0.80 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT20' --mode 'test' --nms 0.95 -b 1 -d 1 --fp16 --fuse

# For DanceTrack validation
python YOLOX/detect.py --dataset 'Dance' --mode 'val' --nms 0.80 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'Dance' --mode 'val' --nms 0.95 -b 1 -d 1 --fp16 --fuse
```

2. FastReID
``` 
# For MOT17 validation
python FastReID/ext_feats.py --dataset 'MOT17' --mode 'val' --nms 0.80
python FastReID/ext_feats.py --dataset 'MOT17' --mode 'val' --nms 0.95

# For MOT17 test
python FastReID/ext_feats.py --dataset 'MOT17' --mode 'test' --nms 0.80  
python FastReID/ext_feats.py --dataset 'MOT17' --mode 'test' --nms 0.95

# For MOT20 validation
python FastReID/ext_feats.py --dataset 'MOT20' --mode 'val' --nms 0.80
python FastReID/ext_feats.py --dataset 'MOT20' --mode 'val' --nms 0.95

# For MOT20 test
python FastReID/ext_feats.py --dataset 'MOT20' --mode 'test' --nms 0.80  
python FastReID/ext_feats.py --dataset 'MOT20' --mode 'test' --nms 0.95

# For DanceTrack validation
python FastReID/ext_feats.py --dataset 'Dance' --mode 'val' --nms 0.80
python FastReID/ext_feats.py --dataset 'Dance' --mode 'val' --nms 0.95

# For DanceTrack test
python FastReID/ext_feats.py --dataset 'Dance' --mode 'test' --nms 0.80  
python FastReID/ext_feats.py --dataset 'Dance' --mode 'test' --nms 0.95
```

3. Tracker
```
# For MOT17 validation
python Tracker/run.py --dataset "MOT17" --mode "val"

# For MOT17 test
python Tracker/run.py --dataset "MOT17" --mode "test"
python gen_test_file.py

# For MOT20 validation
python Tracker/run.py --dataset "MOT20" --mode "val"

# For MOT20 test
python Tracker/run.py --dataset "MOT20" --mode "test"
python gen_test_file.py

# For DanceTrack validation
python Tracker/run.py --dataset "Dance" --mode "val"

# For DanceTrack test
python Tracker/run.py --dataset "Dance" --mode "test"
python gen_test_file.py

```