def build_cost(tracks, dets, args, frame_id, use_reid):
    iou_sim, iou_dist = iou_distance(tracks, dets)
    
    # =========================
    # MAHALANOBIS (SOFT)
    # =========================
    measurements = np.array([d.cxcywh for d in dets])
    maha_cost = np.zeros_like(iou_dist)

    gating_threshold = 9.4877 # tuned from your experiment

    for i, track in enumerate(tracks):

        gating_dist = track.kalman_filter.gating_distance(
            track.mean,
            track.covariance,
            measurements
        )

        # hard reject only extreme outliers
        mask = gating_dist > gating_threshold * 2
        iou_dist[i, mask] = 1.0

        # normalize maha to [0,1]
        maha_norm = np.minimum(gating_dist / gating_threshold, 1.0)

        maha_cost[i] = maha_norm

    # fuse motion
    motion_cost = 0.7 * iou_dist + 0.3 * maha_cost

    
    # =========================
    if use_reid:
        cos_dist = cos_distance(tracks, dets)

        # cost = np.zeros_like(motion_cost)
        # for i, track in enumerate(tracks):

        #     motion_uncertainty = np.trace(track.covariance[:4, :4])
        #     u = motion_uncertainty / (motion_uncertainty + 1e-6)

        #     w_motion = np.exp(-u)
        #     w_reid = 1 - w_motion

        #     cost[i] = w_motion * motion_cost[i] + w_reid * cos_dist[i]
        cost = 0.4 * motion_cost + 0.6 * cos_dist

    else:
        cost = motion_cost

    cost += 0.10 * conf_distance(tracks, dets)
    cost += 0.05 * angle_distance(tracks, dets, frame_id)

    cost[iou_sim <= 0.10] = 1.
    cost = np.clip(cost, 0, 1)

    return cost