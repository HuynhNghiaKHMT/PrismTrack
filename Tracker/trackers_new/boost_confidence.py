import numpy as np
from copy import deepcopy


# =========================================================
# OBJECT → NUMPY ADAPTER
# =========================================================
def dets_to_xyxy(dets):

    # fix logic: too many indices for array: array is 1-dimensional, but 2 were indexed
    # ===========================================
    if len(dets) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    # ===========================================

    arr = []
    for d in dets:
        cx, cy, w, h = d.cxcywh
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        arr.append([x1, y1, x2, y2, d.score])

    return np.array(arr, dtype=np.float32)

def tracks_to_xyxy(tracks, frame_id):

    # fix logic: too many indices for array: array is 1-dimensional, but 2 were indexed
    # ===========================================
    if len(tracks) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    # ===========================================

    arr = []
    for t in tracks:
        cx, cy, w, h = t.cxcywh
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # time_since_update (BoostTrack dùng)
        tsu = frame_id - t.end_frame_id

        arr.append([x1, y1, x2, y2, 0, tsu])
    return np.array(arr)

def update_det_scores(dets, new_scores):
    for i in range(len(dets)):
        dets[i].score = np.float32(new_scores[i])
    return dets


def shape_similarity(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    return shape_similarity_v2(detects, tracks)


def shape_similarity_v2(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))

    return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dh, th)))


def MhDist_similarity(mahalanobis_distance: np.ndarray, softmax_temp: float = 1.0) -> np.ndarray:
    limit = 13.2767
    mahalanobis_distance = deepcopy(mahalanobis_distance)

    mask = mahalanobis_distance > limit
    mahalanobis_distance[mask] = limit

    mahalanobis_distance = limit - mahalanobis_distance

    exp = np.exp(mahalanobis_distance / softmax_temp)
    mahalanobis_distance = exp / exp.sum(0, keepdims=True)

    mahalanobis_distance = np.where(mask, 0, mahalanobis_distance)
    return mahalanobis_distance


def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)

    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) *
        (bboxes1[..., 3] - bboxes1[..., 1]) +
        (bboxes2[..., 2] - bboxes2[..., 0]) *
        (bboxes2[..., 3] - bboxes2[..., 1]) - wh
    )

    return o


def soft_biou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    k1, k2 = 0.25, 0.5
    b2conf = bboxes2[..., 4]

    b1x1 = bboxes1[..., 0] - (bboxes1[..., 2]-bboxes1[..., 0])*(1-b2conf)*k1
    b2x1 = bboxes2[..., 0] - (bboxes2[..., 2]-bboxes2[..., 0])*(1-b2conf)*k2
    xx1 = np.maximum(b1x1, b2x1)

    b1y1 = bboxes1[..., 1] - (bboxes1[..., 3]-bboxes1[..., 1])*(1-b2conf)*k1
    b2y1 = bboxes2[..., 1] - (bboxes2[..., 3]-bboxes2[..., 1])*(1-b2conf)*k2
    yy1 = np.maximum(b1y1, b2y1)

    b1x2 = bboxes1[..., 2] + (bboxes1[..., 2]-bboxes1[..., 0])*(1-b2conf)*k1
    b2x2 = bboxes2[..., 2] + (bboxes2[..., 2]-bboxes2[..., 0])*(1-b2conf)*k2
    xx2 = np.minimum(b1x2, b2x2)

    b1y2 = bboxes1[..., 3] + (bboxes1[..., 3]-bboxes1[..., 1])*(1-b2conf)*k1
    b2y2 = bboxes2[..., 3] + (bboxes2[..., 3]-bboxes2[..., 1])*(1-b2conf)*k2
    yy2 = np.minimum(b1y2, b2y2)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)

    wh = w * h
    o = wh / (
        (b1x2 - b1x1)*(b1y2 - b1y1) +
        (b2x2 - b2x1)*(b2y2 - b2y1) - wh
    )

    return o


def get_mh_dist_matrix(tracks, dets):
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(dets), len(tracks)))

    z = np.array([d.cxcywh for d in dets])
    x = np.array([t.mean[:4] for t in tracks])
    sigma_inv = np.array([
        np.reciprocal(np.diag(t.covariance[:4, :4]))
        for t in tracks
    ])

    return ((z[:, None, :] - x[None, :, :]) ** 2 * sigma_inv[None, :, :]).sum(axis=2)


def bbd_distance(dets, tracks, frame_id, alpha=0.025, beta=0.25, c=1.0):
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(dets), len(tracks))) 

    cost = np.zeros((len(dets), len(tracks)))

    for j, d in enumerate(dets):
        for i, t in enumerate(tracks):
            mean = t.cxcywh
            w, h = mean[2], mean[3]

            dt = np.clip(t.get_delta_tau(frame_id), alpha, beta)

            P = np.array([
                [(c * w) ** 2 * dt, 0],
                [0, (c * h) ** 2 * dt]
            ])

            P_inv = np.linalg.inv(P)

            diff = np.array(d.cxcywh[:2]) - np.array(mean[:2])

            dist = diff.T @ P_inv @ diff
            dist = np.sqrt(dist)

            cost[j, i] = dist 

    return cost


def duo_confidence_boost(dets, tracks, frame_id, iou_limit, det_thresh):
    detections = dets_to_xyxy(dets)
    limit = 13.2767

    if use_bbd: 
        dist = bbd_distance(dets, tracks, frame_id)  
    else:
        dist = get_mh_dist_matrix(tracks, dets) 

    if dist.size > 0:
        min_dist = dist.min(1)

        mask = (min_dist > limit) & (detections[:, 4] < det_thresh)
        boost_dets = detections[mask]
        boost_idx = np.where(mask)[0]

        if len(boost_dets) > 0:
            bdiou = iou_batch(boost_dets, boost_dets) - np.eye(len(boost_dets))
            bdiou_max = bdiou.max(axis=1)

            remaining = boost_idx[bdiou_max <= iou_limit]

            args = np.where(bdiou_max > iou_limit)[0]
            for i in args:
                tmp = np.where(bdiou[i] > iou_limit)[0]
                group = np.append(boost_idx[args], boost_idx[tmp])
                group = np.append(group, boost_idx[i])

                conf_max = np.max(detections[group, 4])
                if detections[boost_idx[i], 4] == conf_max:
                    remaining = np.append(remaining, boost_idx[i])

            mask2 = np.zeros(len(detections), dtype=bool)
            mask2[remaining] = True

            detections[:, 4] = np.where(mask2, det_thresh + 1e-4, detections[:, 4])

    return update_det_scores(dets, detections[:, 4])


def dlo_confidence_boost(dets, tracks,  frame_id, use_rich_s, use_sb, use_vt, dlo_boost_coef, det_thresh ):
    detections = dets_to_xyxy(dets)
    trackers = tracks_to_xyxy(tracks, frame_id)
    bbd_thress = 13.2767

    if len(trackers) == 0 or len(detections) == 0:
        return dets

    if use_bbd: 
        bbd_dist = bbd_distance(dets, tracks, frame_id)
        bbd_sim = np.exp(-bbd_dist / 4)
        bbd_sim[bbd_dist > bbd_thress] = 0
        dist = bbd_sim
    else:
        mh = MhDist_similarity(get_mh_dist_matrix(tracks, dets), 1.0)
        dist = mh

    if use_rich_s:
        sbiou = soft_biou_batch(detections, trackers)
        shape = shape_similarity(detections, trackers)

        S = (dist + shape + sbiou) / 3
    else:
        S = iou_batch(detections, trackers)

    if not use_sb and not use_vt:
        max_s = S.max(1)
        coef = dlo_boost_coef
        detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

    else:
        if use_sb:
            max_s = S.max(1)
            alpha = 0.65
            detections[:, 4] = np.maximum(
                detections[:, 4],
                alpha * detections[:, 4] + (1 - alpha) * max_s ** 1.5
            )

        if use_vt:
            threshold_s = 0.95
            threshold_e = 0.8
            n_steps = 20

            alpha = (threshold_s - threshold_e) / n_steps

            tsu = trackers[:, 5]
            th = np.maximum(threshold_s - tsu * alpha, threshold_e)

            tmp = (S > th).max(1)

            scores = deepcopy(detections[:, 4])
            scores[tmp] = np.maximum(scores[tmp], det_thresh + 1e-5)

            detections[:, 4] = scores

    return update_det_scores(dets, detections[:, 4])


use_dlo_boost = True
use_duo_boost = True
use_bbd = True
use_rich_s = True
use_sb = True
use_vt = True
dlo_boost_coef = 0.65 # 0.5 for MOT20
det_thresh = 0.6 # 0.4 for MOT20
iou_limit = 0.3 # 0.2

def apply_boost(dets, tracks, frame_id, use_dlo_boost, use_duo_boost):

    if len(dets) == 0:
        return dets
    
    if use_dlo_boost: 
        dets = dlo_confidence_boost(
            dets, tracks, frame_id,
            use_rich_s, use_sb, use_vt,
            dlo_boost_coef, det_thresh
        )


    if use_duo_boost: 
        dets = duo_confidence_boost(
            dets, tracks, frame_id, iou_limit, det_thresh
        )

    return dets