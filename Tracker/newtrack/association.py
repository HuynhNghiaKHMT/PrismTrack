import numpy as np
from newtrack.utils import *
from newtrack.kalman_filter_hybird import *

# =========================================================
# Hungarian (Linear Association)
# =========================================================
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b

def build_cost_stage1(tracks, dets, frame_id, use_reid):
    """
    C_final = W_1 * C_IoU + W_2 * C_Appr
    """

    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)

    # Appearance cost
    if use_reid:
        # cosine
        alpha = 0.5
        cos_dist = cos_distance(tracks, dets)
        cost = alpha * iou_dist + (1 - alpha) * cos_dist 
    else:
        cost = iou_dist
        

    # MHIoU/ DIoU gate
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)

def build_cost_stage2(tracks, dets, frame_id):
    """
    C_final = W_1 * C_IoU + W_2 * C_Vel + W_3 * Conf
    """

    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)
    cost = iou_dist

    cost += 0.1 * conf_distance_linear(tracks, dets)
    cost += 0.05 * angle_distance(tracks, dets, frame_id, 3)

    # MHIoU/ DIoU gate
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)

def build_cost_stage3(tracks, dets, frame_id):
    """
    C_final = C_IoU
    """

    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)
    cost = iou_dist

    # MHIoU/ DIoU gate
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)

# =========================================================
# Track-Perspective (Iterative Association)
# =========================================================
def associate(cost, match_thr):
    # Initialization
    matches = []

    # Run
    if cost.shape[0] > 0 and cost.shape[1] > 0:
        # Get index for minimum similarity
        min_ddx = np.argmin(cost, axis=1)
        min_tdx = np.argmin(cost, axis=0)

        # Match tracks with detections
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and cost[tdx, ddx] < match_thr:
                matches.append([tdx, ddx])

    return matches

def build_cost_stage12(tracks, dets_high, dets_low, penalty_p, frame_id, use_reid):
    """
    C_final = W_1 * C_IoU +  W_2 * C_Appr + W_3 * C_Vel +  W_4 * C_Conf
    """

    dets = dets_high + dets_low 
    
    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)
    cos_dist = cos_distance(tracks, dets)

    # Appearance cost
    if use_reid:
        alpha = 0.45
        cost = alpha * iou_dist + (1.0 - alpha) * cos_dist
    else:
        cost = cos_dist

    # Confidence/ Velocity cost
    # cost += 0.05 * conf_distance_linear(tracks, dets)
    cost += 0.1 * angle_distance(tracks, dets, frame_id, 3)

    # Penalty for dets_low, give priority to dets_high. Although the IoU of the Low-conf may be slightly higher.
    cost[:, len(dets_high):] += penalty_p 

    # Gating
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)

def iterative_assignment(cost, match_thr, reduce_step, tracks, dets):
    
    # Iterative Association
    # Lower the threshold to continue searching if there is no longer a match with any of these objects.
    matches = []
    temp_match_thr = match_thr
    while temp_match_thr > 0:
        matches_ = associate(cost, temp_match_thr)
        if len(matches_) == 0: 
            temp_match_thr -= reduce_step
            continue
            
        matches += matches_
        for t, d in matches_:
            cost[t, :] = 1.0  
            cost[:, d] = 1.0

    m_tracks = [t for t, _ in matches]
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = [d for _, d in matches]
    u_dets = [d for d in range(len(dets)) if d not in m_dets]

    return matches, u_tracks, u_dets
