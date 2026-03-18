import numpy as np
from trackers_new.utils import *
from trackers_new.kalman_filter_hybird import *


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

            d[invalid] = 1.0  # hoặc INF nếu bạn muốn hard gating

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

def diou_distance(tracks, dets):
    if len(tracks) == 0 or len(dets) == 0:
        shape = (len(tracks), len(dets))
        return np.zeros(shape), np.zeros(shape)  

    diou_dist = np.zeros((len(tracks), len(dets)))
    diou_sim = np.zeros((len(tracks), len(dets)))

    for i, t in enumerate(tracks):
        tx, ty, tw, th = t.cxcywh
        t_x1, t_y1 = tx - tw/2, ty - th/2
        t_x2, t_y2 = tx + tw/2, ty + th/2

        for j, d in enumerate(dets):
            dx, dy, dw, dh = d.cxcywh
            d_x1, d_y1 = dx - dw/2, dy - dh/2
            d_x2, d_y2 = dx + dw/2, dy + dh/2

            # IoU
            inter_x1 = max(t_x1, d_x1)
            inter_y1 = max(t_y1, d_y1)
            inter_x2 = min(t_x2, d_x2)
            inter_y2 = min(t_y2, d_y2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            area_t = tw * th
            area_d = dw * dh
            union = area_t + area_d - inter_area + 1e-6

            iou = inter_area / union

            # center distance
            center_dist = (tx - dx)**2 + (ty - dy)**2

            # enclosing box
            c_x1 = min(t_x1, d_x1)
            c_y1 = min(t_y1, d_y1)
            c_x2 = max(t_x2, d_x2)
            c_y2 = max(t_y2, d_y2)

            c_diag = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + 1e-6

            diou = iou - center_dist / c_diag

            diou_sim[i, j] = diou          
            diou_dist[i, j] = 1 - diou     

    diou_dist = np.clip(diou_dist, 0, 1)

    return diou_sim, diou_dist

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
    return np.asarray(matches), unmatched_a, unmatched_b


# def build_cost_stage1(tracks, dets, frame_id, use_reid):
#     """
#     C_final = 0.5 * C_HMIoU + 0.5 * C_Appr (nếu có ReID)
#     """

#     alpha = 0.5
#     iou_sim, iou_dist = iou_distance(tracks, dets)
#     diou_sim, diou_dist = diou_distance(tracks, dets)


#     if use_reid:
#         cos_dist = cos_distance(tracks, dets)
#         cost = alpha * iou_dist + (1.0 - alpha) * cos_dist 

#     else:
#         cost = iou_dist


#     # DIoU gate
#     cost[diou_sim <= 0.1] = 1.0

#     ## IoU gate
#     # cost[iou_sim <= 0.1] = 1.0

#     return np.clip(cost, 0, 1)


# def build_cost_stage1(tracks, dets, frame_id, use_reid):
#     alpha = 0.5
#     lambda_ = 0.98
#     INF = 1e5

#     # IoU
#     iou_sim, iou_dist = iou_distance(tracks, dets)
#     diou_sim, diou_dist = diou_distance(tracks, dets)

#     # Appearance
#     if use_reid:
#         # cosine
#         cos_dist = cos_distance(tracks, dets)

#         # Mahalanobis
#         maha_cost = Mahalanobis_distance(tracks, dets, diou_dist)

#         # Combine cost
#         cost = alpha * iou_dist + (1- alpha) * (lambda_ * cos_dist + (1 - lambda_) * maha_cost)
#         # cost = alpha * diou_dist + (1 - alpha) * cos_dist

#         # Kalman gating
#         for i in range(len(tracks)):
#             invalid = maha_cost[i] > 1.0  
#             cost[i, invalid] = 1.0

#     else:
#         cost = diou_dist  # fallback


    # # IoU gating (optional)
    # cost[diou_sim <= 0.1] = 1.0

    # return np.clip(cost, 0, 1)

def build_cost_stage1(tracks, dets, frame_id, use_reid):
    alpha = 0.5
    lambda_ = 0.98
    INF = 1e5

    # IoU
    iou_sim, iou_dist = diou_distance(tracks, dets)

    # Appearance
    if use_reid:
        # cosine
        cos_dist = cos_distance(tracks, dets)

        # BBD
        bbd_cost = bbd_distance(tracks, dets, frame_id)

        # cost = alpha * iou_dist + (1 - alpha) * (lambda_*cos_dist + (1-lambda_)*bbd_cost)
        cost = alpha * iou_dist + (1 - alpha) * cos_dist 

        # ===== REID GATING =====
        bbd_threshold = 9.4877

        for i in range(len(tracks)):
            for j in range(len(dets)):

                # BBD gate
                if bbd_cost[i, j] >= bbd_threshold:
                    cost[i, j] = INF
                    continue


    else:
        cost = iou_dist  

    
    # IoU gating (optional)
    cost[iou_sim <= 0.1] = INF

    return np.clip(cost, 0, 1)

def build_cost_stage2(tracks, dets):
    """
    C_final = 1 - HMIoU
    """
    INF = 1e5

    # MHIoU cost
    iou_sim, iou_dist = iou_distance(tracks, dets)
    # cost = iou_dist

    # DIoU similarity
    diou_sim, diou_dist  = diou_distance(tracks, dets)
    cost = diou_dist

    # DIoU gate
    cost[diou_sim <= 0.1] = INF
    
    # # IoU gate
    # cost[iou_sim <= 0.1] = INF

    return np.clip(cost, 0, 1)