import lap
import numpy as np

def iou_distance(a_tracks, b_tracks):
    """
    Return:
        iou_sim: IoU similarity matrix
        iou_dist: 1 - IoU
    """
    a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)

    # Calculate IoU distance
    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        iou_sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float64)
        iou_dist = 1 - iou_sim
        return iou_sim, iou_dist

    bboxes1 = np.expand_dims(a_boxes, 1)  # (N,1,4)
    bboxes2 = np.expand_dims(b_boxes, 0)  # (1,M,4)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1 )
    h = np.maximum(0.0, yy2 - yy1 )
    inter = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0] ) * (bboxes1[..., 3] - bboxes1[..., 1] )
    area2 = (bboxes2[..., 2] - bboxes2[..., 0] ) * (bboxes2[..., 3] - bboxes2[..., 1] )

    union = area1 + area2 - inter + 1e-6

    # Calculate IoU
    iou_sim = inter / union
    iou_dist = 1.0 - iou_sim
    iou_dist = np.clip(iou_dist, 0.0, 1.0)

    return iou_sim, iou_dist

def giou_distance(a_tracks, b_tracks):
    """
    Return:
        giou_sim: GIoU similarity matrix
        giou_dist: 1 - GIoU
    """
    a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)

    # Calculate GIoU distance
    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        giou_sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float32)
        giou_dist = 1 - giou_sim
        return giou_sim, giou_dist

    bboxes1 = np.expand_dims(a_boxes, 1)  # (N,1,4)
    bboxes2 = np.expand_dims(b_boxes, 0)  # (1,M,4)

    # Intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1 + 1.0)
    h = np.maximum(0.0, yy2 - yy1 + 1.0)
    inter = w * h

    # Areas & IoU
    area1 = (bboxes1[..., 2] - bboxes1[..., 0] + 1.0) * (bboxes1[..., 3] - bboxes1[..., 1] + 1.0)
    area2 = (bboxes2[..., 2] - bboxes2[..., 0] + 1.0) * (bboxes2[..., 3] - bboxes2[..., 1] + 1.0)
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    # Enclosing box
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    wc = np.maximum(0.0, xxc2 - xxc1)
    hc = np.maximum(0.0, yyc2 - yyc1)
    area_enclose = wc * hc + 1e-6

    # Calculate GIoU
    giou = iou - (area_enclose - union) / area_enclose

    giou_sim = giou
    giou_dist = 1.0 - giou_sim
    giou_dist = np.clip(giou_dist, 0.0, 1.0)


    return giou_sim, giou_dist

def ciou_distance(a_tracks, b_tracks):
    """
    Return:
        ciou_sim: CIoU similarity matrix
        ciou_dist: 1 - CIoU
    """
    a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)

    # Calculate CIoU distance
    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        ciou_sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float32)
        ciou_dist = 1 - ciou_sim
        return ciou_sim, ciou_dist

    bboxes1 = np.expand_dims(a_boxes, 1)  # (N,1,4)
    bboxes2 = np.expand_dims(b_boxes, 0)  # (1,M,4)

    # Intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    # Areas & IoU
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    # Centers
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Enclosing box
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + 1e-6

    # Aspect ratio term
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    h1 = h1 + 1.0
    h2 = h2 + 1.0

    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4.0 / (np.pi ** 2)) * (arctan ** 2)

    S = 1.0 - iou
    alpha = v / (S + v + 1e-6)

    # Calculate CIoU
    ciou = iou - inner_diag / outer_diag - alpha * v

    ciou_sim = ciou
    ciou_dist = 1.0 - ciou_sim
    ciou_dist = np.clip(ciou_dist, 0.0, 1.0)

    return ciou_sim, ciou_dist

def diou_distance(a_tracks, b_tracks):
    """
    Return:
        diou_sim: DIoU similarity matrix
        diou_dist: 1 - DIoU
    """
    a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)

    # Calculate DIoU distance
    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        diou_sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float32)
        diou_dist = 1 - diou_sim
        return diou_sim, diou_dist

    bboxes1 = np.expand_dims(a_boxes, 1)  # (N,1,4)
    bboxes2 = np.expand_dims(b_boxes, 0)  # (1,M,4)

    # Intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    # Areas & IoU
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    # Centers
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Enclosing box
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + 1e-6

    # Calculate DIoU
    diou = iou - inner_diag / outer_diag

    diou_sim = diou
    diou_dist = 1.0 - diou_sim
    diou_dist = np.clip(diou_dist, 0.0, 1.0)

    return diou_sim, diou_dist

def hmiou_distance(a_tracks, b_tracks):
    """
    - "+1" pixel convention (inclusive box)
    - NO clamp for height overlap
    Return:
        iou_sim: IoU similarity matrix 
        hmiou_dist: 1 - hiou * iou_sim
    """
    a_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in b_tracks], dtype=np.float64)

    # Calculate IoU distance
    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        hmiou_sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float32)
        hmiou_dist = 1 - hmiou_sim
        return hmiou_sim, hmiou_dist

    bboxes1 = np.expand_dims(a_boxes, 1)  # (N,1,4)
    bboxes2 = np.expand_dims(b_boxes, 0)  # (1,M,4)

    # Calculate HIoU  
    # Height overlap ratio (NO clamp)
    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    hiou = (yy12 - yy11) / (yy22 - yy21 + 1e-6)

    # IoU with "+1" convention
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1 + 1.0 )
    h = np.maximum(0.0, yy2 - yy1 + 1.0 )
    inter = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0] + 1.0 ) * (bboxes1[..., 3] - bboxes1[..., 1] + 1.0 )
    area2 = (bboxes2[..., 2] - bboxes2[..., 0] + 1.0 ) * (bboxes2[..., 3] - bboxes2[..., 1] + 1.0 )
    union = area1 + area2 - inter + 1e-6

    iou = inter / union

    # Calculate HMIoU
    iou_sim = iou
    hmiou_dist = 1.0 - hiou * iou_sim
    hmiou_dist = np.clip(hmiou_dist, 0.0, 1.0)

    return iou_sim, hmiou_dist
    
def wmiou_distance(a_tracks, b_tracks):
    """
    - "+1" pixel convention (inclusive box)
    - NO clamp for height overlap
    Return:
        iou_sim: IoU similarity matrix
        wmiou_dist: 1 - wiou * iou_sim
    """
    a_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in b_tracks], dtype=np.float64)

    # Calculate IoU distance 
    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        iou_sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float32)
        wmiou_dist = 1 - iou_sim
        return iou_sim, wmiou_dist

    bboxes1 = np.expand_dims(a_boxes, 1)  # (N,1,4)
    bboxes2 = np.expand_dims(b_boxes, 0)  # (1,M,4)

    # Calculate HIoU  
    # Weight overlap ratio (NO clamp)
    xx11 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    xx12 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])

    xx21 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    xx22 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])

    wiou = (xx12 - xx11) / (xx22 - xx21 + 1e-6)

    # IoU with "+1" convention
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1 + 1.0 )
    h = np.maximum(0.0, yy2 - yy1 + 1.0 )
    inter = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0] + 1.0 ) * (bboxes1[..., 3] - bboxes1[..., 1] + 1.0 )
    area2 = (bboxes2[..., 2] - bboxes2[..., 0] + 1.0 ) * (bboxes2[..., 3] - bboxes2[..., 1] + 1.0 )
    union = area1 + area2 - inter + 1e-6

    iou = inter / union

    # Calculate WMIoU
    iou_sim = iou
    wmiou_dist = 1.0 - wiou * iou_sim
    wmiou_dist = np.clip(wmiou_dist, 0.0, 1.0)

    return iou_sim, wmiou_dist

def cos_distance(tracks, dets):
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Check logic crash nếu feat=None
    if tracks[0].feat is None or dets[0].feat is None:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)
    
    # Calculate cosine distance
    t_feat = np.concatenate([t.feat for t in tracks], axis=0)
    d_feat = np.concatenate([d.feat for d in dets], axis=0)
    cos_dist = np.clip(1 - np.dot(t_feat, d_feat.T), a_min=0., a_max=1.)

    return cos_dist

def conf_distance_kf(tracks, dets):
    # Calculate confidence similarity
    # t_score = np.array([t.score for t in tracks])
    t_score = np.array([t.score for t in tracks])
    d_score = np.array([d.score for d in dets])
    conf_dist = np.abs(t_score[:, None] - d_score[None, :])

    return conf_dist

def conf_distance_linear(tracks, dets):
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get previous scores
    t_score_prev = []
    for t in tracks:
        frame_ids = sorted(list(t.history.keys()), reverse=True)
        frame_id = frame_ids[min(1, len(frame_ids) - 1)]
        t_score_prev.append(t.history[frame_id][1])

    # Linear projection
    t_score_prev = np.array(t_score_prev)
    t_score = np.array([t.score for t in tracks])
    t_score += (t_score - t_score_prev)

    # Calculate confidence similarity
    d_score = np.array([d.score for d in dets])
    conf_dist = np.abs(t_score[:, None] - d_score[None, :])

    return conf_dist

def get_prev_box(history, frame_id, dt):
    # Try
    target_key = frame_id - dt
    if target_key in history.keys():
        return history[target_key][0]

    # If there are no recent observation
    return history[max(history.keys())][0]

def get_vel_t_d(b_1, b_2):
    # Expand boxes
    b_1, b_2 = b_1[:, np.newaxis, :], b_2[np.newaxis, :, :]

    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_lb = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 3:4]**2) + 1e-5
    norm_rt = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_rb = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 3:4]**2) + 1e-5

    # Get velocities
    vel_lt = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_lt
    vel_lb = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 3] - b_1[:, :, 3]], axis=-1) / norm_lb
    vel_rt = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_rt
    vel_rb = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 3] - b_1[:, :, 3]], axis=-1) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=2)

def calc_angle(vel_t, vel_t_d):
    angle_ = 0
    for vdx in range(vel_t.shape[2]):
        # Divide & Repeat
        vel_t_x = np.repeat(vel_t[:, :, vdx, 0], vel_t_d.shape[1], axis=1)
        vel_t_y = np.repeat(vel_t[:, :, vdx, 1], vel_t_d.shape[1], axis=1)

        # Calculate angle, Normalize to range (0 ~ 1)
        angle = vel_t_x * vel_t_d[:, :, vdx, 0] + vel_t_y * vel_t_d[:, :, vdx, 1]
        angle = np.abs(np.arccos(np.clip(angle, a_min=-1, a_max=1))) / np.pi
        angle_ += angle / 4

    return angle_

def angle_distance(tracks, dets, frame_id, d_t=3):
    # Initialization
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get velocity between track and detections
    track_boxes = np.stack([get_prev_box(t.history, frame_id, d_t) for t in tracks], axis=0)
    vel_t_d = get_vel_t_d(track_boxes, np.stack([d.x1y1x2y2 for d in dets], axis=0))

    # Get angle distance
    angle_dist = calc_angle(np.stack([t.velocity for t in tracks], axis=0)[:, np.newaxis], vel_t_d)

    # Fuse score
    scores = np.array([d.score for d in dets])[np.newaxis, :]
    # angle_dist *= scores

    return angle_dist

def shape_similarity(a_tracks, b_tracks):

    a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)

    if a_boxes.shape[0] == 0 or b_boxes.shape[0] == 0:
        shape_sim = np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float64)
        shape_dist = 1 - shape_sim
        return shape_sim, shape_dist

    dw = (a_boxes[:, 2] - a_boxes[:, 0]).reshape((-1, 1))
    dh = (a_boxes[:, 3] - a_boxes[:, 1]).reshape((-1, 1))
    tw = (b_boxes[:, 2] - b_boxes[:, 0]).reshape((1, -1))
    th = (b_boxes[:, 3] - b_boxes[:, 1]).reshape((1, -1))

    shape_sim = np.exp(
        -(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dh, th))
    )
    shape_dist = 1 - shape_sim

    return shape_sim, shape_dist

def bbd(tracks, dets, frame_id, alpha=0.025, beta=0.25, c=1.0):

    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))

    bbd_thress = 13.2767
    bbd_dist = np.zeros((len(tracks), len(dets)))  # FIX SHAPE

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
            diff = np.array(d.cxcywh[:2]) - np.array(mean[:2])

            dist = diff.T @ P_inv @ diff
            dist = np.sqrt(dist)

            # Normalize (quan trọng)
            dist = min(dist / bbd_thress, 1.0)

            bbd_dist[i, j] = dist

    return bbd_dist

def mhd(tracks, dets):
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))

    z = np.array([d.cxcywh for d in dets])           # (M,4)
    x = np.array([t.mean[:4] for t in tracks])       # (N,4)

    sigma_inv = np.array([
        np.reciprocal(np.diag(t.covariance[:4, :4]) + 1e-6)
        for t in tracks
    ])  # (N,4)

    # (N,M,4)
    diff = z[None, :, :] - x[:, None, :]

    mh = (diff ** 2 * sigma_inv[:, None, :]).sum(axis=2)  # (N,M)

    # normalize
    limit = 13.2767
    mh = np.minimum(mh, limit) / limit

    return mh

# def hmiou_distance(a_tracks, b_tracks):
#     """
#     Return:
#         iou_sim: IoU similarity matrix
#         hmiou_dist: 1 - HIoU * IOU
#     """
#     a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
#     b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)

#     # Calculate IoU distance
#     if len(a_boxes) == 0 or len(b_boxes) == 0:
#         sim = np.zeros((len(a_boxes), len(b_boxes)))
#         return sim, 1 - sim

#     b1 = a_boxes[:, None, :]
#     b2 = b_boxes[None, :, :]

#     # Calculate HIoU # ---- height ratio (NO clamp, NO epsilon) ----
#     h_iou = (np.minimum(b1[..., 3], b2[..., 3]) - np.maximum(b1[..., 1], b2[..., 1])) 
#           / \ (np.maximum(b1[..., 3], b2[..., 3]) - np.minimum(b1[..., 1], b2[..., 1]))

#     # ---- IoU EXACT bbox_overlaps style ----
#     iw = np.minimum(b1[..., 2], b2[..., 2]) - np.maximum(b1[..., 0], b2[..., 0]) + 1
#     ih = np.minimum(b1[..., 3], b2[..., 3]) - np.maximum(b1[..., 1], b2[..., 1]) + 1

#     valid = (iw > 0) & (ih > 0)

#     area1 = (b1[..., 2] - b1[..., 0] + 1) * (b1[..., 3] - b1[..., 1] + 1)
#     area2 = (b2[..., 2] - b2[..., 0] + 1) * (b2[..., 3] - b2[..., 1] + 1)

#     inter = iw * ih
#     union = area1 + area2 - inter

#     iou = np.zeros_like(inter)
#     iou[valid] = inter[valid] / union[valid]

#     # ---- FINAL ----
#     sim = iou                     
#     dist = 1 - h_iou * iou       

#     return sim, dist
