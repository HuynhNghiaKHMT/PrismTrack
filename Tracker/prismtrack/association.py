import numpy as np
from prismtrack.utils import *
from prismtrack.kalman_filter_hybird import *

iou_functions = {
    "iou": iou_distance,
    "ciou": ciou_distance,
    "diou": diou_distance,
    "giou": giou_distance,
    "hmiou": hmiou_distance,
    "wmiou": wmiou_distance,
}

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

def build_cost_stage1(tracks, dets, asso, frame_id, use_reid):
    """
    C_final = W_1 * C_IoU + W_2 * C_Appr
    """

    # MHIoU/ DIoU cost
    overlap = iou_functions.get(asso, iou_distance)
    iou_sim, iou_dist = overlap(tracks, dets)

    # Velocity / Confidence / Shape cost
    motion = iou_dist.copy()


    # Appearance cost
    if use_reid:
        alpha = 0.45
        app = cos_distance(tracks, dets)
        cost = alpha * motion + (1.0 - alpha) * app
    else:
        cost = motion

    # Gating
    cost[iou_sim <= 0.1] = 1e5

    return np.clip(cost, 0, 1)

def build_cost_stage2(tracks, dets, asso, frame_id):
    """
    C_final = W_1 * C_IoU + W_2 * C_Vel + W_3 * Conf
    """

    # MHIoU/ DIoU cost
    overlap = iou_functions.get(asso, iou_distance)
    iou_sim, iou_dist = overlap(tracks, dets)

    # Velocity / Confidence / Shape cost
    cost = iou_dist.copy()

    # Gating
    cost[iou_sim <= 0.1] = 1e5

    return np.clip(cost, 0, 1)

def build_cost_stage3(tracks, dets, asso, frame_id):
    """
    C_final = C_IoU
    """

    # MHIoU/ DIoU cost
    overlap = iou_functions.get(asso, iou_distance)
    iou_sim, iou_dist = overlap(tracks, dets)

    # Velocity / Confidence / Shape cost
    cost = iou_dist.copy()


    # Gating
    cost[iou_sim <= 0.1] = 1e5

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

def build_cost_stage12(tracks, dets_high, dets_low, asso, penalty_p, wm, wv, wc, ws, frame_id, use_reid, dt):
    """
    C_motion = C_IoU +  W_1 * C_Appr + W_2 * C_Vel +  W_3 * C_Conf + W_4 * C_Shape
    C_final = W_1 * C_motion + W_2 * C_Appr
    """

    dets = dets_high + dets_low
    
    # Overlap cost
    overlap = iou_functions.get(asso)
    iou_sim, iou_dist = overlap(tracks, dets)

    # Velocity / Confidence / Shape cost
    motion = iou_dist.copy()
    motion += wv * angle_distance(tracks, dets, frame_id, 3)
    motion += ws * shape_similarity(tracks, dets)[1]
    motion += wc * conf_distance_linear(tracks, dets)

    #Appearance cost
    if use_reid:
        app = cos_distance(tracks, dets)
        cost = wm * motion + (1.0 - wm) * app
    else:
        cost = motion


    # Penalty for dets_low, give priority to dets_high. Although the IoU of the Low-conf may be slightly higher.
    cost[:, len(dets_high):] += penalty_p

    # Gating
    cost[iou_sim <= 0.1] = 1e5

    return np.clip(cost, 0, 1)

def build_cost_stage12_adapt(tracks, dets_high, dets_low, asso, penalty_p, wm, wv, wc, ws, frame_id, use_reid, dt):
    """
    C_motion = C_IoU +  W_1 * C_Appr + W_2 * C_Vel +  W_3 * C_Conf + W_4 * C_Shape
    C_final = W_1 * C_motion + W_2 * C_Appr
    """

    dets = dets_high + dets_low 
    
    # MHIoU/ DIoU cost
    overlap = iou_functions.get(asso)
    iou_sim, iou_dist = overlap(tracks, dets)
    cos_dist = cos_distance(tracks, dets)

    # Velocity / Confidence / Shape cost
    iou_dist += wv * angle_distance(tracks, dets, frame_id, 3)
    iou_dist += wc * conf_distance_linear(tracks, dets)
    iou_dist += ws * shape_similarity(tracks, dets)[1]

    # c1
    adaptive_alpha = max(0.25, 0.45 * np.exp(-0.1 * (dt - 1)))
    # C2
    # alpha_base = 0.45
    # tau = 10.0
    # adaptive_alpha = alpha_base * np.exp(-(dt - 1) / tau)
    # adaptive_alpha = max(0.25, adaptive_alpha)
    # C3
    # tau = 10.0 
    # adaptive_alpha = 0.45 * (1 - 0.6 * (dt - 1) / (dt + tau))
    # adaptive_alpha = max(0.3, adaptive_alpha)
    # C4
    # dt_factor = np.log2(max(5, dt) / 5.0)
    # adaptive_alpha = max(0.3, 0.45 - 0.05 * dt_factor)

    # Appearance cost
    if use_reid:
        cost = adaptive_alpha * iou_dist + (1.0 - adaptive_alpha) * cos_dist
    else:
        cost = cos_dist


    # Penalty for dets_low, give priority to dets_high. Although the IoU of the Low-conf may be slightly higher.
    # dynamic_penalty = penalty_p * (1 + 0.2 * dt_factor)
    cost[:, len(dets_high):] += penalty_p 

    # Gating
    # C1
    # iou_thr = max(0.01, 0.1 * np.exp(-0.15 * (dt - 1)))
    # C2
    iou_thr = 0.1 / (1 + np.log(dt))
    # C3
    # iou_thr = 0.1 / (1 + 0.15 * (dt - 1))
    # C4
    # iou_thr = max(0.01, 0.1 / (1 + 0.5 * dt_factor))
    cost[iou_sim <= iou_thr] = 1.0

    return np.clip(cost, 0, 1)

