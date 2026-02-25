import os
import torch
import pickle
import argparse
import random
import time
from trackers_tracktrack.tracker_exp import Tracker
from utils.etc import *
from trackers_tracktrack.utils import *
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from utils.gbi import gb_interpolation

def make_parser():
    parser = argparse.ArgumentParser("TrackTrack Experiment")
    # Basic Path Args (Giữ nguyên từ file gốc)
    parser.add_argument("--pickle_dir", type=str, default="outputs/2. det_feat/")
    parser.add_argument("--output_dir", type=str, default="outputs/3. track/tracktrack/")
    parser.add_argument("--data_dir", type=str, default="dataset/")
    parser.add_argument("--dataset", type=str, default="MOT17")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--seed", type=float, default=10000)

    # Experiment Flags (Thêm theo yêu cầu của bạn)
    parser.add_argument("--assi", type=str, default="local")   # global / local
    parser.add_argument("--asso", type=str, default="joint")   # multi / joint
    parser.add_argument("--ddel", type=str, default="true")     # use Ddel

    parser.add_argument("--cmc", type=str, default="true")
    parser.add_argument("--reid", type=str, default="true")
    parser.add_argument("--tai", type=str, default="true")
    parser.add_argument("--aflink", type=str, default="true")
    parser.add_argument("--gbi", type=str, default="true")

    # Tracking Hyperparameters (Giữ nguyên từ file gốc)
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--max_time_lost", type=float, default=30)
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)
    
    # Bổ sung các tham số thường được set_parameters gọi
    parser.add_argument("--det_thr", type=float, default=0.6) # track_thresh 0.6 + 0.1
    parser.add_argument("--init_thr", type=float, default=0.7)
    parser.add_argument("--match_thr", type=float, default=0.9)

    return parser

def str2bool(v):
    return v.lower() == "true"

def track_experiment(args, detections, detections_95, data_path, result_folder, mode):
    use_cmc = str2bool(args.cmc)
    use_reid = str2bool(args.reid)

    total_time, total_count = 0, 0
    
    for vid_name in detections.keys():
        # Set proper parameters (Cập nhật pickle_path, data_path cho từng video)
        set_parameters(args, vid_name, mode)

        # Set max time lost
        seq_info = open(data_path + vid_name + '/seqinfo.ini', mode='r')
        for s_i in seq_info.readlines():
            if 'frameRate' in s_i:
                args.max_time_lost = int(s_i.split('=')[-1]) * 2
            if 'imWidth' in s_i:
                args.img_w = int(s_i.split('=')[-1])
            if 'imHeight' in s_i:
                args.img_h = int(s_i.split('=')[-1])

        tracker = Tracker(args, vid_name)
        results = []

        for frame_id in detections[vid_name].keys():
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                               
                track_results = tracker.update(
                    detections[vid_name][frame_id], 
                    detections_95[vid_name][frame_id], 
                    use_cmc=use_cmc, 
                    use_reid=use_reid
                )
            else:
                track_results = tracker.update_without_detections()
            
            total_time += time.time() - start
            total_count += 1

            # Filter results (Giữ nguyên logic gốc)
            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                # Check aspect ratio
                if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                    continue
                # Check track id, min area
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)

            results.append([frame_id, track_ids, x1y1whs, scores])

        # Write kết quả thô
        result_filename = os.path.join(result_folder, f'{vid_name}.txt')
        write_results(result_filename, results)

    return total_time, total_count

def run():
    # Khởi tạo AFLink (Giữ nguyên gốc)
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

    # Đọc detection (Sử dụng pickle_path đã được set_parameters cập nhật hoặc từ dir)
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)
    with open(args.pickle_path_95, 'rb') as f:
        detections_95 = pickle.load(f)

    # Chạy Tracking
    total_time, total_count = track_experiment(args, detections, detections_95, args.data_path, result_folder, args.mode)

    # Post-processing (Tích hợp Flags)
    print('Running post-processing...')
    use_aflink = str2bool(args.aflink)
    use_gbi = str2bool(args.gbi)

    for result_file in os.listdir(result_folder):
        if not result_file.endswith(".txt"): continue
        
        path_in = os.path.join(result_folder, result_file)
        path_out = os.path.join(post_folder, result_file)

        # Mặc định copy sang post nếu không chạy post-processing
        import shutil
        shutil.copy(path_in, path_out)

        # Link (AFLink)
        # if use_aflink and 'Dance' in args.dataset:
        if use_aflink: # change
            linker = AFLink(path_in=path_out, path_out=path_out, model=model, dataset=aflink_dataset,
                            thrT=(0, 20), thrS=100, thrP=0.05)
            linker.link()

        # Gaussian Interpolation (GBI)
        # if use_gbi and 'MOT' in args.dataset:
        if use_gbi: # change
            gb_interpolation(path_out, path_out, interval=30, tau=12)

    # Đánh giá (Evaluation)
    if args.mode == 'val':
        print('Evaluating...')
        evaluate(args, trackers_to_eval + '_post', args.dataset)

    print(f"FPS: {total_count / total_time:.2f}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    os.environ["PYTHONHASHSEED"] = str(int(args.seed))

    run()