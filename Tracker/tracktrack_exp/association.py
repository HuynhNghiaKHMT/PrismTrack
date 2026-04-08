import numpy as np
from tracktrack_exp.utils import *

# =========================
# GLOBAL (Hungarian)
# =========================
def global_assignment(cost, match_thr):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    matches, u_tracks, u_dets = linear_assignment(cost, thresh=match_thr)

    matches = matches.tolist()
    u_tracks = list(u_tracks)
    u_dets = list(u_dets)

    return matches, u_tracks, u_dets

# =========================
# LOCAL (TPA)
# =========================
def local_assignment(cost, match_thr, reduce_step):
    matches = []
    temp_thr = match_thr

    while True:
        matches_ = associate(cost, temp_thr)
        if len(matches_) == 0:
            break

        matches += matches_

        for t, d in matches_:
            cost[t, :] = 1.
            cost[:, d] = 1.

        temp_thr -= reduce_step

    m_tracks = [t for t, _ in matches]
    m_dets = [d for _, d in matches]

    u_tracks = [i for i in range(cost.shape[0]) if i not in m_tracks]
    u_dets = [i for i in range(cost.shape[1]) if i not in m_dets]

    return matches, u_tracks, u_dets


# =========================
# SAFE TRACKS FOR COST BUILDING
# =========================
def safe_tracks_for_cost(tracks):
    safe_tracks = []
    for t in tracks:
        # Lọc bỏ các giá trị None trong history
        clean_history = {k: v for k, v in t.history.items() if v is not None}
        
        # Tạo một object tạm thời có các thuộc tính mà utils.py cần
        class SafeTrack:
            def __init__(self, original_track, clean_hist):
                self.history = clean_hist
                self.velocity = original_track.velocity
                self.score = original_track.score
                self.feat = original_track.feat
                self.x1y1x2y2 = original_track.x1y1x2y2
                self.kalman_filter = original_track.kalman_filter
                self.mean = original_track.mean
                self.covariance = original_track.covariance
        
        safe_tracks.append(SafeTrack(t, clean_history))

    return safe_tracks

def mahalanobis_gatting(tracks, dets, iou_dist):

    if len(tracks) > 0 and len(dets) > 0:

        measurements = np.array([d.cxcywh for d in dets])

        gating_threshold = 9.4877  # chi2inv95[4]

        for t, track in enumerate(tracks):

            maha_dist = track.kalman_filter.gating_distance(
                track.mean,
                track.covariance,
                measurements
            )

            invalid = maha_dist > gating_threshold
            iou_dist[t, invalid] = 1.0

        return iou_dist

# =========================
# BUILD COST MATRIX
# =========================
def build_cost_stage1(tracks, dets, frame_id):

    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))

    iou_sim, iou_dist = iou_distance(tracks, dets)
    cos_dist = cos_distance(tracks, dets)

    cost = 0.5 * iou_dist + 0.5 * cos_dist

    # # Mahalanobis gating
    # measurements = np.array([d.cxcywh for d in dets])
    # theta_mhd = 9.4877

    # for t, track in enumerate(tracks):
    #     maha = track.kalman_filter.gating_distance(
    #         track.mean,
    #         track.covariance,
    #         measurements
    #     )
    #     cost[t, maha > theta_mhd] = 1.0

    # cost += 0.05 * angle_distance(tracks, dets, frame_id)

    cost[iou_sim <= 0.05] = 1.0
    cost = np.clip(cost, 0, 1)

    return cost

def build_cost_stage2(tracks, dets, frame_id):

    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))

    iou_sim, iou_dist = iou_distance(tracks, dets)

    cost = iou_dist.copy()

    # cost += 0.10 * conf_distance(tracks, dets)
    # cost += 0.10 * angle_distance(tracks, dets, frame_id)

    cost[iou_sim <= 0.05] = 1.0
    cost = np.clip(cost, 0, 1)

    return cost

def multi_stage_assignment(tracks, dets_high, dets_low, dets_del, args, frame_id, use_reid, assi_type):
    matches_all = []
    used_dets = set()
    used_tracks = set()

    def run_stage(curr_tracks, dets, thr, stage):
        if len(curr_tracks) == 0 or len(dets) == 0:
            return [], list(range(len(curr_tracks))), list(range(len(dets)))

        curr_tracks = safe_tracks_for_cost(curr_tracks)

        if stage == 1:
            cost = build_cost_stage1(curr_tracks, dets_high, frame_id)
        elif stage == 2:
            cost = build_cost_stage2(curr_tracks, dets_low, frame_id)

        if assi_type == "global":
            return global_assignment(cost, thr)
        else:
            return local_assignment(cost, thr, args.reduce_step)

    # =========================
    # STAGE 1: HIGH
    # =========================
    matches, u_tracks, u_dets = run_stage(tracks, dets_high, args.match_thr, stage=1)

    for t, d in matches:
        matches_all.append((t, d))
        used_tracks.add(t)
        used_dets.add(d)

    # =========================
    # STAGE 2: LOW
    # =========================
    remain_tracks = [tracks[i] for i in u_tracks]

    matches2, u_tracks2, u_dets2 = run_stage(remain_tracks, dets_low, 0.5, stage=2)

    for t, d in matches2:
        matches_all.append((u_tracks[t], d + len(dets_high)))

    # =========================
    # STAGE 3: DEL
    # =========================
    if args.ddel.lower() == "true":

        remain_tracks = [remain_tracks[i] for i in u_tracks2]

        matches3, u_tracks3, u_dets3 = run_stage(remain_tracks, dets_del, 0.5, stage=3)

        for t, d in matches3:
            matches_all.append((
                u_tracks[u_tracks2[t]],
                d + len(dets_high) + len(dets_low)
            ))

    # =========================
    # BUILD FINAL OUTPUT
    # =========================
    all_dets = dets_high + dets_low + dets_del

    matched_t = [t for t, _ in matches_all]
    matched_d = [d for _, d in matches_all]

    u_tracks_final = [i for i in range(len(tracks)) if i not in matched_t]
    u_dets_final = [i for i in range(len(all_dets)) if i not in matched_d]

    return matches_all, u_tracks_final, u_dets_final, all_dets


def build_cost(tracks, dets, args, frame_id, use_reid):
    
    # iou_sim, iou_dist = iou_distance(tracks, dets)
    tracks = safe_tracks_for_cost(tracks) # change
    iou_sim, iou_dist = iou_distance(tracks, dets) # change
   
    if use_reid:
        cos_dist = cos_distance(tracks, dets)
        cost = 0.5* iou_dist + 0.5 * cos_dist

    else:
        cost = iou_dist

    cost += 0.10 * conf_distance(tracks, dets)
    cost += 0.05 * angle_distance(tracks, dets, frame_id)

    cost[iou_sim <= 0.10] = 1.
    cost = np.clip(cost, 0, 1)

    return cost

def joint_assignment(tracks, dets_high, dets_low, dets_del, args, frame_id, use_reid, assi_type):
    
    if args.ddel.lower() != "true":
        dets_del = []

    dets = dets_high + dets_low + dets_del

    cost = build_cost(tracks, dets, args, frame_id, use_reid)


    # penalty
    # cost[:, len(dets_high):len(dets_high + dets_low)] += args.penalty_p
    # cost[:, len(dets_high + dets_low):] += args.penalty_q
    cost[:, len(dets_high):len(dets_high) + len(dets_low)] += args.penalty_p
    cost[:, len(dets_high) + len(dets_low):] += args.penalty_q

    if assi_type == "global":
        matches, u_tracks, u_dets = global_assignment(cost, args.match_thr)
    else:
        matches, u_tracks, u_dets = local_assignment(cost, args.match_thr, args.reduce_step)

    return matches, u_tracks, u_dets, dets


# =========================
# MAIN ASSOCIATION SWITCH
# =========================

def run_association(tracks, dets_high, dets_low, dets_del, args, frame_id, use_reid):

    if args.asso == "joint":
        return joint_assignment(
            tracks, dets_high, dets_low, dets_del, args, frame_id, use_reid, args.assi)

    else:  # multi
        return multi_stage_assignment(
            tracks, dets_high, dets_low, dets_del, args, frame_id, use_reid, args.assi)
