import numpy as np
from copy import deepcopy
from newtrack.utils import *

"""
Script modified from BoostTrack++: 
https://github.com/vukasin-stanojevic/BoostTrack
"""

chi4inv95 = {
    1: 7.7794,
    2: 9.4877,
    3: 11.1433,
    4: 9.4877,
    5: 13.2767,
    6: 18.4668
}

def dets_to_xyxy(dets):

    # fix logic: too many indices for array: array is 1-dimensional, but 2 were indexed
    if len(dets) == 0:
        return np.zeros((0, 5), dtype=np.float64)

    arr = [[d.cxcywh[0] - d.cxcywh[2]/2, d.cxcywh[1] - d.cxcywh[3]/2, 
            d.cxcywh[0] + d.cxcywh[2]/2, d.cxcywh[1] + d.cxcywh[3]/2, 
            d.score] for d in dets]
    
    return np.array(arr, dtype=np.float64)

def tracks_to_xyxy(tracks, frame_id):

    # fix logic: too many indices for array: array is 1-dimensional, but 2 were indexed
    if len(tracks) == 0:
        return np.zeros((0, 6), dtype=np.float64)

    arr = [[t.cxcywh[0] - t.cxcywh[2]/2, t.cxcywh[1] - t.cxcywh[3]/2, 
            t.cxcywh[0] + t.cxcywh[2]/2, t.cxcywh[1] + t.cxcywh[3]/2, 
            0.0, frame_id - t.end_frame_id] for t in tracks]
    
    return np.array(arr, dtype=np.float64)

def soft_biou_distance(a_tracks, b_tracks, frame_id):

    a_boxes = dets_to_xyxy(a_tracks)
    b_boxes = tracks_to_xyxy(b_tracks, frame_id)

    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float64)
        return sim, 1.0 - sim
    
    bboxes1 = np.expand_dims(a_boxes, 1)
    bboxes2 = np.expand_dims(b_boxes, 0)
    
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
    inter = w * h

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    union = area1 + area2 - inter + 1e-6

    iou_sim = inter / union
    iou_dist = 1 - iou_sim

    return iou_sim, iou_dist

def mhd_matrix(dets, tracks):
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(dets), len(tracks)))

    z = np.array([d.cxcywh for d in dets])
    x = np.array([t.mean[:4] for t in tracks])
    sigma_inv = np.array([
        np.reciprocal(np.diag(t.covariance[:4, :4]))
        for t in tracks
    ])

    return ((z[:, None, :] - x[None, :, :]) ** 2 * sigma_inv[None, :, :]).sum(axis=2)

def mahalanobis_similarity(mh_dist, softmax_temp = 1.0):
    limit = 13.2767
    dist = np.copy(mh_dist)

    mask = dist > limit
    dist[mask] = limit

    dist = limit - dist

    exp = np.exp(dist / softmax_temp)
    dist = exp / exp.sum(0, keepdims=True)

    dist = np.where(mask, 0, dist)
    return dist

def bbd_matrix(dets, tracks, alpha = 0.025, beta = 0.35, c = 1.0):
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(dets), len(tracks)))

    det_xy = np.array([d.cxcywh[:2] for d in dets], dtype=np.float64)
    trk_xywh = np.array([t.cxcywh for t in tracks], dtype=np.float64)

    trk_xy = trk_xywh[:, :2]

    w = trk_xywh[:, 2]
    h = trk_xywh[:, 3]

    dt = np.array([np.clip(max(1, t.time_since_update), alpha, beta) for t in tracks])

    inv_x = 1.0 / (((c * w) ** 2) * dt)
    inv_y = 1.0 / (((c * h) ** 2) * dt)

    dx = det_xy[:, None, 0] - trk_xy[None, :, 0]
    dy = det_xy[:, None, 1] - trk_xy[None, :, 1]

    return (np.sqrt(dx * dx * inv_x[None, :] + dy * dy * inv_y[None, :]))

def bbox_based_similarity(bbd_dist):
    limit = 13.2767
    tau = 4.0
    sim = np.exp(-bbd_dist / tau)
    sim[bbd_dist > limit] = 0

    return sim

def mahalanobis_distance(dets, tracks, softmax_temp = 1.0):

    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(dets), len(tracks))), np.zeros((len(dets), len(tracks)))

    z = np.array([d.cxcywh for d in dets])
    x = np.array([t.mean[:4] for t in tracks])
    sigma_inv = np.array([
        np.reciprocal(np.diag(t.covariance[:4, :4]))
        for t in tracks
    ])

    dist =  ((z[:, None, :] - x[None, :, :]) ** 2 * sigma_inv[None, :, :]).sum(axis=2)
    mhd_dist = np.copy(dist)

    limit = 13.2767
    mask = dist > limit
    dist[mask] = limit
    dist = limit - dist

    exp = np.exp(dist / softmax_temp)
    mhd_sim = exp / exp.sum(0, keepdims=True)
    mhd_sim = np.where(mask, 0, mhd_sim)

    return mhd_sim, mhd_dist

def bbox_based_distance(dets, tracks, alpha = 0.025, beta = 0.35, c = 1.0):

    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(dets), len(tracks))), np.zeros((len(dets), len(tracks)))

    bbd_thresh = 13.2767

    det_xy = np.array([d.cxcywh[:2] for d in dets], dtype=np.float64)
    trk_xywh = np.array([t.cxcywh for t in tracks], dtype=np.float64)

    trk_xy = trk_xywh[:, :2]

    w = trk_xywh[:, 2]
    h = trk_xywh[:, 3]

    # dt = np.array([np.clip(t.get_delta_tau(frame_id), alpha, beta) for t in tracks])
    dt = np.array([np.clip(max(1, t.time_since_update), alpha, beta) for t in tracks])

    inv_x = 1.0 / (((c * w) ** 2) * dt)
    inv_y = 1.0 / (((c * h) ** 2) * dt)

    dx = det_xy[:, None, 0] - trk_xy[None, :, 0]
    dy = det_xy[:, None, 1] - trk_xy[None, :, 1]

    bbd_dist = np.sqrt(dx * dx * inv_x[None, :] + dy * dy * inv_y[None, :])

    tau = 4.0
    bbd_sim = np.exp(-bbd_dist / tau)
    bbd_sim[bbd_dist > bbd_thresh] = 0

    return bbd_sim, bbd_dist

def update_det_scores(dets, new_scores):
    for i in range(len(dets)):
        dets[i].score = np.float32(new_scores[i])
    return dets

# def cos_sim(tracks, dets):
#     return 1 - cos_distance(tracks, dets).T

# def angle_sim(tracks, dets, frame_id):
#     return  1 - angle_distance(tracks, dets, frame_id).T

# def score_sim(tracks, dets):
#     return 1 - conf_distance_linear(tracks, dets).T

def dlo_confidence_boost(dets, tracks, dlo_boost_coef, det_thresh, frame_id, use_rich_s, use_sb, use_vt, use_bbd):
    detections = dets_to_xyxy(dets)
    trackers = tracks_to_xyxy(tracks, frame_id)

    if len(trackers) == 0 or len(detections) == 0:
        return dets

    if use_bbd: 
        # bbd_dist = bbd_matrix(dets, tracks)
        # dist_sim = bbox_based_similarity(bbd_dist)
        dist_sim = bbox_based_distance(dets, tracks)[0]
    else:
        # mhd_dist = mhd_matrix(dets, tracks)
        # dist_sim = mahalanobis_similarity(mhd_dist)
        dist_sim = mahalanobis_distance(dets, tracks)[0]

    if use_rich_s:
        # sbiou_sim, sbiou_dist = soft_biou_distance(dets, tracks, frame_id)
        # shape_sim, shape_dist = shape_similarity(detections, trackers)
        sbiou_sim  = soft_biou_distance(dets, tracks, frame_id)[0]
        # shape_sim = shape_similarity(detections, trackers)[0]

        # app_sim = cos_sim(tracks, dets)
        # vel_sim = angle_sim(tracks, dets, frame_id)
        # conf_sim = score_sim(tracks, dets)

        # S = (dist_sim + shape_sim + sbiou_sim) / 3
        S = (sbiou_sim + dist_sim) / 2 
        # S = dist_sim

        # S =  S = (sbiou_sim + dist_sim + app_sim) / 3

    else:
        S = iou_distance(dets, tracks)[0]

    if not use_sb and not use_vt:
        max_s = S.max(1)
        coef = dlo_boost_coef
        detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

    else:
        if use_sb:
            max_s = S.max(1)
            alpha = 0.65
            threshold_q = 1.5
            detections[:, 4] = np.maximum(detections[:, 4], alpha * detections[:, 4] + (1 - alpha) * max_s ** threshold_q)

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

def duo_confidence_boost(dets, tracks, iou_limit, det_thresh, frame_id, use_bbd):
    detections = dets_to_xyxy(dets)

    if len(tracks) == 0:
        return dets

    limit = 13.2767
    iou_thr = 0.1

    if use_bbd: 
        # dist = bbd_matrix(dets, tracks)  
        dist = bbox_based_distance(dets, tracks)[1]
    else:
        # dist = mhd_matrix(dets, tracks) 
        dist = mahalanobis_distance(dets, tracks)[1]


    if dist.size > 0:

        iou_sim = iou_distance(dets, tracks)[0]
        max_ious = iou_sim.max(axis=1)
        min_dist = dist.min(axis=1)

        mask = ((min_dist > limit) & (detections[:, 4] < det_thresh) & (max_ious < iou_thr))
        boost_dets = detections[mask]
        boost_idx = np.argwhere(mask).reshape((-1,))

        if len(boost_dets) > 0:

            boost_det_objs = [dets[i] for i in boost_idx]
            bdiou = (iou_distance(boost_det_objs,boost_det_objs)[0] - np.eye(len(boost_dets)))
            bdiou_max = bdiou.max(axis=1)
            remaining_boxes = boost_idx[bdiou_max <= iou_limit]
            args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))

            for i in range(len(args)):

                boxi = args[i]
                tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                args_tmp = np.append(np.intersect1d(boost_idx[args], boost_idx[tmp]), boost_idx[boxi])
                conf_max = np.max(detections[args_tmp, 4])

                if detections[boost_idx[boxi], 4] == conf_max:
                    remaining_boxes = np.array(remaining_boxes.tolist() + [boost_idx[boxi]])

            mask2 = np.zeros_like(detections[:, 4],dtype=np.bool_)
            mask2[remaining_boxes] = True

            detections[:, 4] = np.where(mask2, det_thresh + 1e-4, detections[:, 4])

    return update_det_scores(dets, detections[:, 4])

def duo_confidence_boost_v2(dets, tracks, iou_limit, det_thresh, frame_id, use_bbd):
    
    detections = dets_to_xyxy(dets)
    
    if len(tracks) == 0:
        return dets
    
    limit = 13.2767
    iou_thr = 0.1
    
    if use_bbd: 
        # dist = bbd_matrix(dets, tracks)  
        dist = bbox_based_distance(dets, tracks)[1]
    else:
        # dist = mhd_matrix(dets, tracks) 
        dist = mahalanobis_distance(dets, tracks)[1]

    # Lấy ma trận khoảng cách và IoU để kiểm tra sự tranh chấp
    iou_sim = iou_distance(dets, tracks)[0]

    if dist.size > 0:
        min_dists = dist.min(1)
        max_ious = iou_sim.max(1)

        # CHIẾN THUẬT PHÒNG THỦ:
        # - min_dists > 13.2767: Quá xa các track cũ
        # - detections[:, 4] < det_thresh: Điểm thấp (cần cứu)
        # - max_ious < 0.1: QUAN TRỌNG - Nếu dính líu (IoU) với track cũ, 
        #   để DLO lo hoặc bỏ qua, DUO không được tạo ID mới ở đây.
        mask = (min_dists > limit) & (detections[:, 4] < det_thresh) & (max_ious < iou_thr)
        
        boost_idx = np.where(mask)[0]
        boost_dets = detections[mask]
        boost_det_objs = [dets[i] for i in boost_idx]

        if len(boost_dets) > 0:
            # NMS nội bộ cực gắt để tránh x2, x3 ID cho cùng một vùng nhiễu
            bdiou = iou_distance(boost_det_objs, boost_det_objs)[0] - np.eye(len(boost_dets))
            bdiou_max = bdiou.max(axis=1)

            # Giữ lại những box đơn độc
            remaining_indices = boost_idx[bdiou_max <= iou_limit]
            
            # Xử lý các cụm chồng lấn: Chỉ chọn thằng tốt nhất
            overlap_args = np.where(bdiou_max > iou_limit)[0]
            processed = set()
            for i in overlap_args:
                if i in processed: continue
                neighbors = np.where(bdiou[i] > iou_limit)[0]
                group = np.append(neighbors, i)
                
                # Trong một cụm nhiễu, chỉ chọn 1 đại diện có score cao nhất
                actual_group_idx = boost_idx[group]
                best_idx = actual_group_idx[np.argmax(detections[actual_group_idx, 4])]
                remaining_indices = np.unique(np.append(remaining_indices, best_idx))
                processed.update(group)

            # Cập nhật điểm số
            final_mask = np.zeros(len(detections), dtype=bool)
            final_mask[remaining_indices.astype(int)] = True
            detections[:, 4] = np.where(final_mask, det_thresh + 1e-4, detections[:, 4])

    return update_det_scores(dets, detections[:, 4])

def IDCBoost(dets, tracks, dlo_boost_coef, det_thresh, iou_limit, frame_id, use_dlo, use_duo):
    """
    STEP 1: DLO (Detection of Likely Objects):
    Objective: Rescue tracked objects that are hidden (low score)
    
    TEP 2: DUO (Detection of Unobserved Objects):
    Objective: Identify newly appearing objects that the detector is not yet confident in.
    IMPORTANT: DUO only works on detections with a score < det_thresh.
    (That is, objects that DLO cannot recover or are not related to the old track)
    """

    if len(dets) == 0:
        return dets
    
    # DLO (Detection of Likely Objects) 
    if use_dlo:
        use_bbd = True
        use_rich_s = True
        use_sb = True
        use_vt = True
        dets = dlo_confidence_boost(
            dets, tracks, 
            dlo_boost_coef, det_thresh, frame_id,
            use_rich_s, use_sb, use_vt, use_bbd
        )

    # DUO (Detection of Unobserved Objects)
    if use_duo:
        use_bbd = False
        dets = duo_confidence_boost_v2(
            dets, tracks, 
            iou_limit, det_thresh, frame_id, 
            use_bbd
        )

    return dets