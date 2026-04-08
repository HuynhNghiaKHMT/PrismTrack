import numpy as np
from newtrack.utils import *
from newtrack.kalman_filter_hybird import *


def Mahalanobis_distance(tracks, dets, iou_dist):
    if len(tracks) > 0 and len(dets) > 0:
        measurements = np.array([d.cxcywh for d in dets])
        maha_cost = np.zeros_like(iou_dist)

        gating_threshold = 9.4877  # chi2inv95[4]
        # gating_threshold = 16.919  # chi2inv95[9]

        for i, track in enumerate(tracks):
            d = track.kalman_filter.gating_distance(
                track.mean,
                track.covariance,
                measurements,
                only_position=False
            )

            # Gating (TRƯỚC khi normalize)
            invalid = d > gating_threshold

            # Normalize
            d = d / gating_threshold
            d = np.clip(d, 0, 1)

            d[invalid] = 1.0  # hoặc 1.0 nếu bạn muốn hard gating

            maha_cost[i] = d

    else:
        maha_cost = np.zeros_like(iou_dist)

    return maha_cost

def bbd_distance(tracks, dets, frame_id, alpha=0.025, beta=0.25, c=1.0):
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))

    cost = np.zeros((len(tracks), len(dets)))

    for i, t in enumerate(tracks):
        mean = t.cxcywh
        w, h = mean[2], mean[3]

        dt = np.clip(t.get_delta_tau(frame_id), alpha, beta)

        P = np.array([
            [(c * w) ** 2 * dt, 0],
            [0, (c * h) ** 2 * dt]
        ])

        P_inv = np.linalg.inv(P)

        for j, d in enumerate(dets):
            z = d.cxcywh

            diff = z[:2] - mean[:2]

            dist = diff.T @ P_inv @ diff
            dist = np.sqrt(dist)  

            cost[i, j] = dist

    return cost

# =========================================================
# Linear (Hungarian Association)
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
    C_final = 0.5 * C_HMIoU + 0.5 * C_Appr (nếu có ReID)
    """
    alpha = 0.5
    lambda_ = 0.98

    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)

    # Appearance cost
    if use_reid:
        # cosine
        cos_dist = cos_distance(tracks, dets)

        # BBD
        bbd_cost = bbd_distance(tracks, dets, frame_id)

        # cost = alpha * iou_dist + (1 - alpha) * (lambda_*cos_dist + (1-lambda_)*bbd_cost)
        cost = alpha * iou_dist + (1 - alpha) * cos_dist 

        # ===== REID GATING =====
        bbd_threshold = 16

        # for i in range(len(tracks)):
        #     for j in range(len(dets)):

        #         # BBD gate
        #         if bbd_cost[i, j] >= bbd_threshold:
        #             cost[i, j] = 1.0
        #             continue
        
        # cost += 0.1 * conf_distance_kf(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, 3) # HMIoU
        # cost += 0.05 * angle_distance(tracks, dets, frame_id, 3) # DIou
    else:
        cost = iou_dist
        

    # MHIoU/ DIoU gate
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)

def build_cost_stage2(tracks, dets, frame_id):
    """
    C_final = 1 - HMIoU
    """

    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)
    cost = iou_dist

    # cost += 0.25 * conf_distance_linear(tracks, dets) # HMIoU
    # cost += 0.1 * conf_distance_linear(tracks, dets) # DIoU
    cost += 0.1 * conf_distance_linear(tracks, dets)  + 0.05 * angle_distance(tracks, dets, frame_id, 3) # DLO 0.2, no * score

    # MHIoU/ DIoU gate
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)

def build_cost_stage3(tracks, dets):
    """
    C_final = 1 - HMIoU
    """

    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)
    cost = iou_dist

    # MHIoU/ DIoU gate
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)

# =========================================================
# TPA (Track-Perspective Association)
# =========================================================
def tpa_associate(cost, match_thr):
    matches = []

    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return matches

    min_det = np.argmin(cost, axis=1)
    min_track = np.argmin(cost, axis=0)

    for t, d in enumerate(min_det):
        if min_track[d] == t and cost[t, d] < match_thr:
            matches.append([t, d])

    return matches

def tpa_assignment(cost, match_thr, reduce_step = 0.05):

    matches = []
    cost = cost.copy()

    while True:
        new_matches = tpa_associate(cost, match_thr)

        if len(new_matches) == 0:
            break

        matches.extend(new_matches)

        # remove matched rows & cols
        for t, d in new_matches:
            cost[t, :] = 1.0
            cost[:, d] = 1.0

        match_thr -= reduce_step

    matched_t = [t for t, _ in matches]
    matched_d = [d for _, d in matches]

    u_tracks = [i for i in range(cost.shape[0]) if i not in matched_t]
    u_dets   = [i for i in range(cost.shape[1]) if i not in matched_d]

    return matches, u_tracks, u_dets

def build_cost_stage12(tracks, dets_high, dets_low, frame_id, use_reid, penalty_p = 0.2):

    dets = dets_high + dets_low

    # MHIoU/ DIoU cost
    iou_sim, iou_dist = diou_distance(tracks, dets)

    if use_reid:
        cos_dist = cos_distance(tracks, dets)
        cost = 0.5 * iou_dist + 0.5 * cos_dist
        cost += 0.1 * conf_distance_linear(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, 3)
    else:
        cost = iou_dist

    # penalty for los dets 
    # if len(dets_low) > 0:
    #     start = len(dets_high)
    #     cost[:, start:] += penalty_p

    # MHIoU/ DIoU gate
    cost[iou_sim <= 0.1] = 1.0

    return np.clip(cost, 0, 1)