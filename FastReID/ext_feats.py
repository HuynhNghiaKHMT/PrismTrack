import os
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer
from tqdm import tqdm

def make_parser():
    # Initialization
    parser = argparse.ArgumentParser("Track")

    # Data args
    parser.add_argument("--dataset",default="MOT17",type=str,help="dataset for eval")
    parser.add_argument("--mode",default="val",type=str,help="mode for eval")
    parser.add_argument("--nms", default=0.70, type=float, help="test nms threshold")

    parser.add_argument("--data_path", type=str, default="dataset/MOT17/val/")
    parser.add_argument("--pickle_path", type=str, default="outputs/1. det/mot17_val_0.70.pickle")
    parser.add_argument("--output_path", type=str, default="outputs/2. det_feat/mot17_val_0.70.pickle")
    parser.add_argument("--config_path", type=str, default="FastReID/configs/MOT17/sbs_S50.yml")
    parser.add_argument("--weight_path", type=str, default="FastReID/weights/mot17_sbs_S50.pth")

    # Else
    parser.add_argument("--seed", type=float, default=10000)

    return parser


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()
    
    # Auto resolve from dataset & mode
    dataset = args.dataset.lower()
    mode = args.mode.lower()
    nms = f"{args.nms:.2f}"


    split = "train" if mode == "val" else "test"
    half = "half" if mode == "val" else ""

    args.data_path   = f"dataset/{args.dataset}/{split}/"
    args.pickle_path = f"outputs/1. det/{dataset}_{mode}_{nms}.pickle"
    args.output_path = f"outputs/2. det_feat/{dataset}_{mode}_{nms}.pickle"
    args.config_path = f"FastReID/configs/{args.dataset}_{half}/sbs_S50.yml"
    args.weight_path = f"FastReID/weights/{dataset}_{half}_sbs_S50.pth"

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Get encoder
    embedder = EmbeddingComputer(config_path=args.config_path, weight_path=args.weight_path)

    # Read detection
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Progress bar
    total_frames = sum(len(detections[vid]) for vid in detections)
    pbar = tqdm(total=total_frames, desc="Extracting", unit="frame")

    # Feature extraction
    for vid_name in detections.keys():
        for frame_id in detections[vid_name].keys():
            # If there is no detection
            if detections[vid_name][frame_id] is None:
                continue

            # Read image
            if 'MOT' in args.data_path:
                img = cv2.imread(args.data_path + vid_name + '/img1/%06d.jpg' % frame_id)
            else:
                img = cv2.imread(args.data_path + vid_name + '/img1/%08d.jpg' % frame_id)

            # Get detection
            detection = detections[vid_name][frame_id]

            # Get features
            if detection is not None:
                embedding = embedder.compute_embedding(img, detection[:, :4])
                detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)

            # Logging
            pbar.update(1)

    pbar.close()

    embedder.print_summary()

    # Save
    with open(args.output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
