## Model Zoo
Save weights files under "./weights/"
  - [mot17_half.pth.tar](https://drive.google.com/file/d/1R-eMf5SgwmizMkOjqJq3ZiurWBNGYf1j/view?usp=drive_link)
  - [mot17.pth.tar](https://drive.google.com/file/d/1MAb-Bhikx-fWe0VlJON_VMrYIyyyrt-F/view?usp=drive_link)
  - [mot20_half.pth.tar](https://drive.google.com/file/d/1H1BxOfinONCSdQKnjGq0XlRxVUo_4M8o/view?usp=drive_link)
  - [mot20.pth.tar](https://drive.google.com/file/d/1FunATdHrWfK95RiiEIw2GJ-gXB-tXMPB/view?usp=drive_link)
  - [dance.pth.tar](https://drive.google.com/file/d/1ZKpYmFYCsRdXuOL60NRuc7VXAFYRskXB/view?usp=drive_link)

## Training
Trained weights will be created under "./weights/"
```

```

## Detection

Detection results will be created under "../outputs/[1. det](https://drive.google.com/drive/folders/14bTBqCVOXr-mWxVKEn58BWfTmm2paG3U?usp=sharing)/" as pickle files
```
python YOLOX/detect.py --dataset 'MOT17' --mode 'val' --nms 0.70 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT17' --mode 'test' --nms 0.70 -b 1 -d 1 --fp16 --fuse

# For MOT20
python YOLOX/detect.py --dataset 'MOT20' --mode 'val' --nms 0.70 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'MOT20' --mode 'test' --nms 0.70 -b 1 -d 1 --fp16 --fuse

# For DanceTrack 
python YOLOX/detect.py --dataset 'Dance' --mode 'val' --nms 0.70 -b 1 -d 1 --fp16 --fuse
python YOLOX/detect.py --dataset 'Dance' --mode 'test' --nms 0.70 -b 1 -d 1 --fp16 --fuse
```

## Reference
  - https://github.com/Megvii-BaseDetection/YOLOX
  - https://github.com/ifzhang/ByteTrack
