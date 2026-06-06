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

# def build_cost_stage12(tracks, dets_high, dets_low, penalty_p, wm, wv, wc, ws, frame_id, use_reid, dt):
#     """
#     C_motion = C_IoU +  W_1 * C_Appr + W_2 * C_Vel +  W_3 * C_Conf + W_4 * C_Shape
#     C_final = W_1 * C_motion + W_2 * C_Appr
#     """

#     dets = dets_high + dets_low 
    
#     # MHIoU/ DIoU cost
#     iou_sim, iou_dist = hmiou_distance(tracks, dets)

#     # Velocity / Confidence / Shape cost
#     motion = iou_dist.copy()
#     motion += wv * angle_distance(tracks, dets, frame_id, 3)
#     motion += wc * conf_distance_linear(tracks, dets)
#     motion += ws * shape_similarity(tracks, dets)[1]
#     # motion += 0.2 * bbd(tracks, dets,frame_id) 

#     # Appearance cost
#     if use_reid:
#         alpha = wm
#         app = cos_distance(tracks, dets)
#         cost = alpha * motion + (1.0 - alpha) * app
#     else:
#         cost = motion

#     # Penalty for dets_low, give priority to dets_high. Although the IoU of the Low-conf may be slightly higher.
#     cost[:, len(dets_high):] += penalty_p 

#     # Gating
#     cost[iou_sim <= 0.1] = 1e5


#     return np.clip(cost, 0, 1)


# association.py

def compute_aw_weight(app_sim, base_weight, max_diff=0.5):
    """
    app_sim: Ma trận độ tương đồng diện mạo (1 - cos_distance)
    base_weight: Trọng số cơ sở (ví dụ: 1.0 - alpha)
    max_diff: Tham số aw_param
    """
    # app_sim nên là Similarity (càng cao càng giống), không phải Distance
    sim_matrix = 1.0 - app_sim 
    
    w_bonus = np.zeros_like(sim_matrix)

    # Thưởng theo hàng (Tracklet so với các Detections)
    if sim_matrix.shape[1] >= 2:
        for i in range(sim_matrix.shape[0]):
            inds = np.argsort(-sim_matrix[i])
            # Bonus = Hiệu của top 1 và top 2
            row_weight = min(sim_matrix[i, inds[0]] - sim_matrix[i, inds[1]], max_diff)
            w_bonus[i] += row_weight / 2

    # Thưởng theo cột (Detection so với các Tracklets)
    if sim_matrix.shape[0] >= 2:
        for j in range(sim_matrix.shape[1]):
            inds = np.argsort(-sim_matrix[:, j])
            col_weight = min(sim_matrix[inds[0], j] - sim_matrix[inds[1], j], max_diff)
            w_bonus[:, j] += col_weight / 2

    return base_weight + w_bonus

def build_cost_stage12(tracks, dets_high, dets_low, penalty_p, wm, wv, wc, ws, frame_id, use_reid, dt):
    dets = dets_high + dets_low 
    
    # 1. Tính Motion Cost (giữ nguyên logic của bạn)
    iou_sim, iou_dist = hmiou_distance(tracks, dets)
    motion = iou_dist.copy()
    motion += wv * angle_distance(tracks, dets, frame_id, 3)
    motion += wc * conf_distance_linear(tracks, dets)
    motion += ws * shape_similarity(tracks, dets)[1]
    # motion += 0.2 * bbd(tracks, dets,frame_id) 

    # 2. Tính Appearance Cost với Adaptive Weighting
    if use_reid:
        app_dist = cos_distance(tracks, dets)
        
        # Bước AW: Tính toán trọng số linh hoạt thay vì dùng alpha cố định
        # wm ở đây đóng vai trò là alpha (trọng số cho motion)
        # (1.0 - wm) là trọng số cơ sở cho appearance
        adaptive_app_weight = compute_aw_weight(app_dist, base_weight=(1.0 - wm), max_diff=0.7)
        
        # C_final = W_motion * Motion_Dist + W_adaptive_app * App_Dist
        cost = wm * motion + adaptive_app_weight * app_dist
        # cost = (1 - adaptive_app_weight)  * motion + adaptive_app_weight * app_dist
    else:
        cost = motion

    # 3. Penalty và Gating (giữ nguyên)
    cost[:, len(dets_high):] += penalty_p 
    cost[iou_sim <= 0.1] = 1e5

    return np.clip(cost, 0, 1)