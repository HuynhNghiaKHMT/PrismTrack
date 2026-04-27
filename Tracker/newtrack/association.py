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

def build_cost_stage12(tracks, dets_high, dets_low, penalty_p, wm, wv, wc, ws, frame_id, use_reid, dt):
    """
    C_motion = C_IoU +  W_1 * C_Appr + W_2 * C_Vel +  W_3 * C_Conf + W_4 * C_Shape
    C_final = W_1 * C_motion + W_2 * C_Appr
    """

    dets = dets_high + dets_low 
    
    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)
    cos_dist = cos_distance(tracks, dets)

    # t_score = np.array([t.score for t in tracks])
    # d_score = np.array([d.score for d in dets]) 
    # conf = t_score[:, None] * d_score[None, :]
    # conf = 1 - conf_distance_kf(tracks, dets) 

    # Velocity / Confidence / Shape cost
    iou_dist += wv * angle_distance(tracks, dets, frame_id, 3) # 0.25
    iou_dist += wc * conf_distance_linear(tracks, dets) # 0.3
    iou_dist += ws * shape_similarity(tracks, dets)[1]
    # iou_dist +=  0.05 *  mhd(tracks, dets) 
    # iou_dist +=  0.2 *  bbd(tracks, dets,frame_id) 

    # Appearance cost
    if use_reid:
        alpha = wm
        cost = alpha * iou_dist + (1.0 - alpha) * cos_dist
    else:
        cost = cos_dist

    # Confidence/ Velocity cost
    # cost += 0.05 * angle_distance(tracks, dets, frame_id, 3)
    # cost += 0.05 * conf_distance_linear(tracks, dets) 
    # cost += 0.2 * shape_similarity(tracks, dets)[1]

    # Penalty for dets_low, give priority to dets_high. Although the IoU of the Low-conf may be slightly higher.
    cost[:, len(dets_high):] += penalty_p 

    # Gating
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)


def build_cost_stage12_adapt(tracks, dets_high, dets_low, penalty_p, frame_id, use_reid, dt):
    """
    C_motion = C_IoU +  W_1 * C_Appr + W_2 * C_Vel +  W_3 * C_Conf + W_4 * C_Shape
    C_final = W_1 * C_motion + W_2 * C_Appr
    """

    dets = dets_high + dets_low 
    
    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)
    cos_dist = cos_distance(tracks, dets)
    iou_dist += 0.15 * angle_distance(tracks, dets, frame_id, 3)
    iou_dist +=  0.1 * shape_similarity(tracks, dets)[1]

    # c1
    # adaptive_alpha = max(0.35, 0.5 * np.exp(-0.1 * (dt - 1)))
    # C2
    alpha_base = 0.45
    tau = 10.0
    adaptive_alpha = alpha_base * np.exp(-(dt - 1) / tau)
    adaptive_alpha = max(0.35, adaptive_alpha)
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

