import numpy as np
from copy import deepcopy

"""
Script modified from BoostTrack++: 
https://github.com/vukasin-stanojevic/BoostTrack
"""

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

def iou_batch(a_tracks, b_tracks):

    # a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    # b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)
    a_boxes = a_tracks
    b_boxes = b_tracks

    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float64)
        return sim, 1.0 - sim

    bboxes1 = np.expand_dims(a_boxes, 1)
    bboxes2 = np.expand_dims(b_boxes, 0)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0] ) * (bboxes1[..., 3] - bboxes1[..., 1] )
    area2 = (bboxes2[..., 2] - bboxes2[..., 0] ) * (bboxes2[..., 3] - bboxes2[..., 1] )
    union = area1 + area2 - inter

    iou_sim = inter / union + 1e-6
    iou_dist = 1 - iou_sim

    return iou_sim

def soft_biou_batch(a_tracks, b_tracks):

    # a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    # b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)
    a_boxes = a_tracks
    b_boxes = b_tracks

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

def shape_similarity(a_tracks, b_tracks):
    if a_tracks.size == 0 or b_tracks.size == 0:
        sim = np.zeros((a_tracks.shape[0], b_tracks.shape[0]), dtype=np.float64)
        return sim, 1.0 - sim

    dw = (a_tracks[:, 2] - a_tracks[:, 0]).reshape((-1, 1))
    dh = (a_tracks[:, 3] - a_tracks[:, 1]).reshape((-1, 1))
    tw = (b_tracks[:, 2] - b_tracks[:, 0]).reshape((1, -1))
    th = (b_tracks[:, 3] - b_tracks[:, 1]).reshape((1, -1))

    shape_sim = np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dh, th)))
    shape_dist = (1 - shape_sim)

    return shape_sim, shape_dist

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

def mahalanobis_distance(mh_dist, softmax_temp = 1.0):
    limit = 13.2767
    dist = np.copy(mh_dist)

    mask = dist > limit
    dist[mask] = limit

    dist = limit - dist

    exp = np.exp(dist / softmax_temp)
    dist = exp / exp.sum(0, keepdims=True)

    dist = np.where(mask, 0, dist)
    return dist

def bbox_based_distance(dets, tracks, frame_id, alpha=0.025, beta=0.25, c=1.0):
    
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(dets), len(tracks))) 

    bbd_thress = 13.2767
    bbd_dist = np.zeros((len(dets), len(tracks)))

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

            bbd_dist[j, i] = dist 

    bbd_sim = np.exp(-bbd_dist / 4)
    bbd_sim[bbd_dist > bbd_thress] = 0

    return bbd_sim, bbd_dist

def update_det_scores(dets, new_scores):
    for i in range(len(dets)):
        dets[i].score = np.float32(new_scores[i])
    return dets

def dlo_confidence_boost(dets, tracks, dlo_boost_coef, det_thresh, frame_id, use_rich_s, use_sb, use_vt, use_bbd):
    detections = dets_to_xyxy(dets)
    trackers = tracks_to_xyxy(tracks, frame_id)
    bbd_thress = 13.2767

    if len(trackers) == 0 or len(detections) == 0:
        return dets

    if use_bbd: 
        bbd_sim, _ = bbox_based_distance(dets, tracks, frame_id)
        dist_sim = bbd_sim
    else:
        mh_dist = get_mh_dist_matrix(tracks, dets)
        dist_sim = mahalanobis_distance(mh_dist, 1.0)

    if use_rich_s:
        sbiou_sim, sbiou_dist = soft_biou_batch(detections, trackers)
        shape_sim, shape_dist = shape_similarity(detections, trackers)

        S = (dist_sim + shape_sim + sbiou_sim) / 3
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
            detections[:, 4] = np.maximum(detections[:, 4], alpha * detections[:, 4] + (1 - alpha) * max_s ** 1.5
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

def duo_confidence_boost(dets, tracks, iou_limit, det_thresh, frame_id, use_bbd):
    detections = dets_to_xyxy(dets)
    limit = 13.2767

    if use_bbd: 
        dist = bbox_based_distance(dets, tracks, frame_id)  
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

def duo_confidence_boost_v2(dets, tracks, iou_limit, det_thresh, frame_id, use_bbd):
    
    detections = dets_to_xyxy(dets)
    
    if len(tracks) == 0:
        return dets
    
    if use_bbd: 
        _, bbd_dist = bbox_based_distance(dets, tracks, frame_id)  
        dist = bbd_dist
    else:
        dist = get_mh_dist_matrix(tracks, dets) 


    # Lấy ma trận khoảng cách và IoU để kiểm tra sự tranh chấp
    iou_sim = iou_batch(detections, tracks_to_xyxy(tracks, frame_id))

    if dist.size > 0:
        min_dists = dist.min(1)
        max_ious = iou_sim.max(1)

        # CHIẾN THUẬT PHÒNG THỦ:
        # - min_dists > 13.2767: Quá xa các track cũ
        # - detections[:, 4] < det_thresh: Điểm thấp (cần cứu)
        # - max_ious < 0.1: QUAN TRỌNG - Nếu dính líu (IoU) với track cũ, 
        #   để DLO lo hoặc bỏ qua, DUO không được tạo ID mới ở đây.
        mask = (min_dists > 13.2767) & (detections[:, 4] < det_thresh) & (max_ious < 0.1)
        
        boost_idx = np.where(mask)[0]
        boost_dets = detections[mask]

        if len(boost_dets) > 0:
            # NMS nội bộ cực gắt để tránh x2, x3 ID cho cùng một vùng nhiễu
            bdiou = iou_batch(boost_dets, boost_dets) - np.eye(len(boost_dets))
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