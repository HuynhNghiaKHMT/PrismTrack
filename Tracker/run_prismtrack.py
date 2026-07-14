import os
import torch
import pickle
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from prismtrack.tracker import Tracker
from prismtrack.utils import *
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from utils.etc_new import *
from utils.gbi import gb_interpolation

"""
Script modified from TrackTrack: 
https://github.com/kamkyu94/TrackTrack
"""

def make_parser():
    parser = argparse.ArgumentParser("TrackTrack Experiment")
    
    # Basic Path Args
    parser.add_argument("--pickle_dir", type=str, default="outputs/2. det_feat/")
    parser.add_argument("--output_dir", type=str, default="outputs/3. track/newtrack/")
    parser.add_argument("--data_dir", type=str, default="dataset/")
    parser.add_argument("--dataset", type=str, default="MOT17")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--seed", type=float, default=10000)

    # Experiment Flags
    parser.add_argument("--kf", type=str, default="new", help="kalman filter")
    parser.add_argument("--cmc", type=str, default="true", help="camera motion compensation")
    parser.add_argument("--idcboost", type=str, default="true", help="improve detection confidence boosting")
    parser.add_argument("--asso", type=str, default="hmiou", help="function: iou/giou/diou/ciou/hmiou/wmiou")
    parser.add_argument("--reid", type=str, default="true", help="re-identify")
    parser.add_argument("--tpa", type=str, default="true", help="track perspective association")
    parser.add_argument("--ttr", type=str, default="true", help="tentative Track Recovery")
    parser.add_argument("--aflink", type=str, default="true", help="Post-processing: Appearance Free Link")
    parser.add_argument("--gbi", type=str, default="true", help="post-processing: Gradient Boosting Interpolation")

    # Tracking Hyperparameters
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track")
    parser.add_argument("--min_ratio", type=float, default=1.6, help="min aspect ratio of bounding box")
    parser.add_argument("--min_box_area", type=float, default=100, help="min area of bounding box")
    parser.add_argument("--max_age", type=float, default=30, help="the frames for keep lost tracks")

    # Mutil-Stage Hyperparameters
    parser.add_argument("--det_high_thr", type=float, default=0.6, help="matching threshold for tracking")
    parser.add_argument("--det_low_thr", type=float, default=0.0, help="matching threshold for tracking")
    parser.add_argument("--det_init_thr", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--match_thr", type=float, default=0.8, help="matching threshold for tracking")

    # IDCBoost Hyperparameters
    parser.add_argument("--iou_limit", type=float, default=0.3, help="matching threshold for tracking")
    parser.add_argument("--det_thr", type=float, default=0.6, help="matching threshold for tracking")
    parser.add_argument("--boost_coef", type=float, default=0.65, help="matching threshold for tracking")

    # Track Perspective Hyperparameters
    parser.add_argument("--penalty_p", type=float, default=0.20, help="penalty for low confidence detections")
    parser.add_argument("--reduce_step", type=float, default=0.05, help="reduce step for iterative association")

    # Weights Hyperparameters
    parser.add_argument("--w_vel", type=float, default=0.15, help="the weight of Velocity in cost matrix")
    parser.add_argument("--w_conf", type=float, default=0.05, help="the weight of Confidence in cost matrix")
    parser.add_argument("--w_shape", type=float, default=0.15, help="the weight of Shape in cost matrix")
    parser.add_argument("--w_motion", type=float, default=0.45, help="the weight of Motion in cost matrix")

    # Low sequence hyperparameters
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hz", type=int, default=30) 
    parser.add_argument("--dt", type=int, default=1) 

    return parser

def str2bool(v):
    return v.lower() == "true"

def track_experiment(args, detections, data_path, result_folder, mode, pbar=None):
    use_cmc = str2bool(args.cmc)
    use_idcboost = str2bool(args.idcboost)
    use_reid = str2bool(args.reid)
    use_tpa = str2bool(args.tpa)
    use_ttr = str2bool(args.ttr)
    total_time, total_count = 0, 0
    
    for vid_name in detections.keys():
        # Set proper parameters
        set_parameters(args, vid_name, mode)

        # Set max time lost
        seq_info = open(data_path + vid_name + '/seqinfo.ini', mode='r')
        for s_i in seq_info.readlines():
            if 'frameRate' in s_i:
                fps = int(s_i.split('=')[-1])   
                args.fps = fps
                args.dt = max(1, int(round(fps / args.hz)))
                args.min_hits = 1 if args.dt != 1 else 3
                args.max_time_lost = int(fps * 2)
                # print(f"{vid_name} fps {fps}")
            if 'imWidth' in s_i:
                args.img_w = int(s_i.split('=')[-1])
            if 'imHeight' in s_i:
                args.img_h = int(s_i.split('=')[-1])

        tracker = Tracker(args, vid_name)
        results = []
        results_dict = {}

        for frame_id in detections[vid_name].keys():
            torch.cuda.synchronize()
            start = time.time()

            det = detections[vid_name][frame_id]

            if (frame_id - 1) % args.dt == 0 and det is not None:
                track_results = tracker.update(det, use_cmc, use_idcboost, use_reid, use_tpa, use_ttr)
            else:
                track_results = tracker.update_without_detections(use_ttr)
            
            torch.cuda.synchronize()
            total_time += time.time() - start
            total_count += 1

            if not use_ttr:
                # Filter results
                x1y1whs, track_ids, scores = [], [], []
                for t in track_results:
                    # Check aspect ratio
                    if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > args.min_ratio:
                        continue
                    # Check track id, min area
                    if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                        x1y1whs.append(t.x1y1wh)
                        track_ids.append(t.track_id)
                        scores.append(t.score)

                results.append([frame_id, track_ids, x1y1whs, scores])
            else:
                # Filter results
                for (f_id, track_id, box, score) in track_results:
                    if f_id not in results_dict:
                        results_dict[f_id] = [[], [], []]

                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1

                    if 'MOT' in data_path and w / h > args.min_ratio:
                        continue

                    if track_id > 0 and w * h > args.min_box_area:
                        results_dict[f_id][0].append(track_id)
                        results_dict[f_id][1].append([x1, y1, w, h])
                        results_dict[f_id][2].append(score)
            
            pbar.update(1)

        if use_ttr:
            results = []
            for f_id in sorted(results_dict.keys()):
                ids, boxes, scores = results_dict[f_id]
                results.append([f_id, ids, boxes, scores])
    
        # Write Results
        result_filename = os.path.join(result_folder, f'{vid_name}.txt')
        write_results(result_filename, results)

    pbar.close()

    return total_time, total_count

def run():
    # initialize AFLink
    model = PostLinker()
    model.load_state_dict(torch.load('Tracker/AFLink/AFLink_epoch20.pth', map_location='cpu'))
    aflink_dataset = LinkData('', '')

    print(f'Running Experiment: {args.dataset} {args.mode}')
    set_parameters(args, args.dataset, args.mode)

    trackers_to_eval = f"{args.dataset.lower()}_{args.mode}"
    result_folder = os.path.join(args.output_dir, trackers_to_eval)
    os.makedirs(result_folder, exist_ok=True)
    post_folder = result_folder + '_post/'
    os.makedirs(post_folder, exist_ok=True)


    if str2bool(args.reid):
        args.pickle_dir = "outputs/2. det_feat/"
    else:
        args.pickle_dir = "outputs/1. det/"

    # Read file detection.pickle 
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Progress bar
    total_frames = sum(len(detections[vid]) for vid in detections)
    pbar = tqdm(total=total_frames, desc="Tracking", unit="frame", dynamic_ncols=True)

    # Run Tracking
    track_time, track_count = track_experiment(args, detections, args.data_path, result_folder, args.mode, pbar)

    # Post-processing
    print('Running post-processing...')
    use_aflink = str2bool(args.aflink)
    use_gbi = str2bool(args.gbi)

    # result_files = [f for f in os.listdir(result_folder)if f.endswith(".txt")]
    # for result_file in tqdm(result_files, desc="Post Processing", unit="seq"):

    torch.cuda.synchronize()
    start_post_time = time.time()

    for result_file in os.listdir(result_folder):
        if not result_file.endswith(".txt"): continue
        
        path_in = os.path.join(result_folder, result_file)
        path_out = os.path.join(post_folder, result_file)

        # By default, copy to post if post-processing is not running.
        import shutil
        shutil.copy(path_in, path_out)

        # Appearance Free Link (AFLink)
        if use_aflink and 'Dance' in args.dataset:
            # print(f"Running AFLink on {result_file}...")
            linker = AFLink(path_in=path_out, path_out=path_out, model=model, dataset=aflink_dataset,
                            thrT=(0, 20), thrS=100, thrP=0.05)
            linker.link()

        # Gradient Boosting Interpolation (GBI)
        if use_gbi and 'MOT' in args.dataset:
            # print(f"Running GBI on {result_file}...")
            gb_interpolation(path_out, path_out, interval=30, tau=12)

    torch.cuda.synchronize()
    post_time = time.time() - start_post_time

    latency_track = (track_time / track_count) * 1000
    speed_track = 1000 / latency_track
    print(f"[*] ONLINE TRACKER ONLY:")
    print(f"  - Latency : {latency_track:.2f} ms/frame")
    print(f"  - Speed   : {speed_track:.2f} FPS")
    
    latency_post = (post_time / track_count) * 1000
    speed_post = 1000 / latency_post
    print(f"[*] OFFLINE POST-PROCESSING ONLY:")
    print(f"  - Latency: {latency_post:.2f} ms/frame")
    print(f"  - Speed: {speed_post:.2f} FPS")
    
    total_latency = latency_track + latency_post
    print(f"[*] TOTAL PIPELINE TRACKER:")
    print(f"  - Latency : {total_latency:.2f} ms/frame")
    print(f"  - Speed: {1000 / total_latency:.2f} FPS")

    # Evaluation
    if args.mode == 'val':
        print('Evaluating...')
        evaluate(args, trackers_to_eval + '_post', args.dataset)


if __name__ == "__main__":

    import tempfile
    print("TEMP DIR:", tempfile.gettempdir())

    args = make_parser().parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    os.environ["PYTHONHASHSEED"] = str(int(args.seed))

    run()