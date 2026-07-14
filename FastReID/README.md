## Model Zoo
Save weights files under "./weights/"
  - [mot17_half_sbs_S50.pth](https://drive.google.com/file/d/1kTG7mVNhYGicR0IXZ0Y1rebVoBRfOMGY/view?usp=drive_link)
  - [mot17_sbs_S50.pth](https://drive.google.com/file/d/1rUYqWIj0nsQ23rDSv8NVx0Rrp3Lco1KP/view?usp=drive_link)
  - [mot20_half_sbs_S50.pth](https://drive.google.com/file/d/1xMI_PpfeY02yfkHzRHZfA4KZtRqHak1o/view?usp=drive_link)
  - [mot20_sbs_S50.pth](https://drive.google.com/file/d/1RhMnTt9JCuZUWk-jPhDPX2NQCZ5g_O3m/view?usp=drive_link)
  - [dance_sbs_S50.pth](https://drive.google.com/file/d/1c9Vn4PADNKFrCuS0HxhPz3PcTvvLWVhc/view?usp=drive_link)


## Training
Trained weights will be created under "./weights/"
```
# For MOT17
python train_net.py --num-gpus 1 --config-file 'configs/MOT17_half/sbs_S50.yml'
python train_net.py --num-gpus 1 --config-file 'configs/MOT17/sbs_S50.yml'

# For MOT20
python train_net.py --num-gpus 1 --config-file 'configs/MOT20_half/sbs_S50.yml'
python train_net.py --num-gpus 1 --config-file 'configs/MOT20/sbs_S50.yml'

# For DanceTrack
python train_net.py --num-gpus 1 --config-file 'configs/DanceTrack/sbs_S50.yml'
```

## Feature Extraction
Detection + feature extraction results will be created under "../outputs/[2. det_feat]()/" as pickle files
```
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

## Reference
  - https://github.com/JDAI-CV/fast-reid
