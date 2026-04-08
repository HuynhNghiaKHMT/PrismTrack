import numpy as np
from newtrack_oru.utils import *
from newtrack_oru.kalman_filter_hybird import *


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
        # cost += 0.05 * angle_distance(tracks, dets, frame_id, 5)  # DLO

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

def build_cost_stage3_for_obs(tracks, dets, track_boxes=None):
    """
    HybridSORT-style Stage 3 cost:
    - Lost track dùng last_observation (nếu có)
    - New track dùng KF state
    - Dựa trên DIoU
    """

    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))

    # =========================
    # Override boxes nếu có last_obs
    # =========================
    if track_boxes is not None:
        # Tạo fake tracks để reuse diou_distance()
        class TmpTrack:
            def __init__(self, box):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                self._cxcywh = np.array([cx, cy, w, h])

            @property
            def cxcywh(self):
                return self._cxcywh

        tracks_for_cost = [TmpTrack(b) for b in track_boxes]
    else:
        tracks_for_cost = tracks

    # =========================
    # DIoU
    # =========================
    iou_sim, iou_dist = diou_distance(tracks_for_cost, dets)
    cost = iou_dist.copy()

    # =========================
    # Gate (Hybrid nhẹ)
    # =========================
    # loosen gate cho LOST (vì dùng last_obs)
    gate_thr = 0.05
    cost[iou_sim <= gate_thr] = 1.0

    return np.clip(cost, 0, 1)