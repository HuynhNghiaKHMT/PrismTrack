# PrismTrack

PrismTrack: Perspective-Aware Multi-Cue Association for Robust Multi-Object Tracking

## Abstract
Multi-Object Tracking (MOT) remains challenging due to object occlusion, complex motions, and detection unreliability in crowded scenarios. This work introduces an enhanced MOT framework is proposed to integrate and optimize state-of-the-art components, specifically Improved Detection Confidence Boost (IDCBoost) and Track-Perspective-Based Association (TPA). The confidence boosting process is refined by introducing Bbox-Based Distance (BBD), a deterministic metric that compensates for the instability of conventional Mahalanobis distance under unreliable Kalman Filter predictions. Moreover, the TPA strategy is enhanced through a comprehensive cost matrix that adaptively fuses spatial overlap, appearance descriptors, velocity direction, and shape consistency. Through the elimination of redundant computational overhead while maintaining robust association logic, a superior balance between accuracy and efficiency is achieved. Extensive experiments demonstrate that the integrated pipeline reaches state-of-the-art performance on MOT17 and achieves highly competitive results within the leading group of Tracking-by-Detection (TBD) methods on both MOT20 and DanceTrack benchmarks.

## Pipeline

<center>
<img src="assets/prismtrack_pipeline.png" width="800"/>
</center>

## Directory structure

```bash
PrismTrack
├── YOLOX/
    └── detect.py
├── FastReID/
    └── ext_feats.py
├── Tracker/
    ├── AFLink/
    ├── CMC/
    ├── prismtrack/
    ├── trackeval/
    ├── utils/
    └── run_prismtrack.py
├── dataset/
    ├── MOT17
    ├── MOT20
    └── DanceTrack
├── outputs/
    ├── 1. det
    ├── 2. det_feat
    └── 3. track
├── assets/ 
├── utils/ 
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```


## Installation
### Step1. Install PrismTrack
```bash
git clone https://github.com/HuynhNghiaKHMT/PrismTrack.git
cd PrismTrack

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Step2. Install Dataset
  - MOT17: https://motchallenge.net/data/MOT17.zip
  - MOT20: https://motchallenge.net/data/MOT20.zip
  - DanceTrack: https://dancetrack.github.io/

### Step3. Install Model Zoo
  - YOLOX: https://github.com/kamkyu94/TrackTrack/tree/main/1.%20YOLOX
  - FastReID: https://github.com/kamkyu94/TrackTrack/tree/main/2.%20FastReID

## Run
### 1. YOLOX
Tracking results will be created under "outputs/1. det/"

```bash
# For MOT17 
python YOLOX/detect.py --dataset 'MOT17' --mode 'val' --nms 0.70 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT17' --mode 'test' --nms 0.70 -b 1 -d 1 --fp16 --fuse

# For MOT20
python YOLOX/detect.py --dataset 'MOT20' --mode 'val' --nms 0.70 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT20' --mode 'test' --nms 0.70 -b 1 -d 1 --fp16 --fuse

# For DanceTrack 
python YOLOX/detect.py --dataset 'Dance' --mode 'val' --nms 0.70 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'Dance' --mode 'test' --nms 0.70 -b 1 -d 1 --fp16 --fuse
```

### 2. FastReID
Tracking results will be created under "outputs/2. det_feat/"

```bash
# For MOT17 
python FastReID/ext_feats.py --dataset 'MOT17' --mode 'val' --nms 0.70
python FastReID/ext_feats.py --dataset 'MOT17' --mode 'test' --nms 0.70  

# For MOT20 
python FastReID/ext_feats.py --dataset 'MOT20' --mode 'val' --nms 0.70
python FastReID/ext_feats.py --dataset 'MOT20' --mode 'test' --nms 0.70  

# For DanceTrack 
python FastReID/ext_feats.py --dataset 'Dance' --mode 'val' --nms 0.70
python FastReID/ext_feats.py --dataset 'Dance' --mode 'test' --nms 0.70  
```

### 3. Tracker
Tracking results will be created under "outputs/3. track/"

```bash
# For MOT17
python Tracker/run_prismtrack.py --dataset "MOT17" --mode "val" 
python Tracker/run_prismtrack.py --dataset "MOT17" --mode "test" 

# For MOT20
python Tracker/run_prismtrack.py --dataset "MOT20" --mode "val" 
python Tracker/run_prismtrack.py --dataset "MOT20" --mode "test" 

# For DanceTrack
python Tracker/run_prismtrack.py --dataset "Dance" --mode "val" 
python Tracker/run_prismtrack.py --dataset "Dance" --mode "test" 

```

### 4. Summit
```bash
python Tracker/utils/gen_test_file.py 
```

## Benchmark Performance
Results on MOT17, MOT20 and DanceTrack challenge test set

| Dataset    | HOTA | IDF1 | MOTA | AssA | DetA | 
| ---------- | ---- | ---- | ---- | -----| -----|
| MOT17      | 00.0 | 00.0 | 00.0 | 00.0 | 00.0 | 
| MOT20      | 00.0 | 00.0 | 00.0 | 00.0 | 00.0 |
| DanceTrack | 00.0 | 00.0 | 00.0 | 00.0 | 00.0 |

## Demo
<img src="assets/demo.gif" alt="demo" style="zoom:34%;" />

## Citation

If you find this work useful, please consider to cite our paper:
```
@article{,
  title={PrismTrack: Perspective-Aware Multi-Cue Association for Robust Multi-Object Tracking},
  author={Huynh Trung Nghia},
  year={2026}
}
```

## Acknowledgement

A large part of the code is borrowed from [TrackTrack](https://github.com/kamkyu94/TrackTrack), [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack), [HybridSORT](https://github.com/ymzis69/HybridSORT). Many thanks for their wonderful works.