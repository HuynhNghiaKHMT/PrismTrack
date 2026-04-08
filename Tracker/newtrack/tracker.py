from trackers.cmc import *
from newtrack.utils import *
from newtrack.track import *
from newtrack.association import *
from newtrack.boost_confidence import apply_boost

class Tracker(object):
    def __init__(self, args, vid_name):
        # Initialize
        self.args = args
        self.max_time_lost = args.max_time_lost

        # Initialize
        self.tracks = []
        self.frame_id = 0
        self.counter = TrackCounter()

        # Set global motion compensation model
        # self.cmc = CMC(vid_name)
        self.cmc = CMC(vid_name) if args.cmc == "true" else None # change

    def init_tracks(self, dets):
        
        use_tai = self.args.tai.lower() == "true"
        
        if use_tai:

            # Get alive tracks, iou_similarity, and scores
            tracks = [t for t in self.tracks if t.state == TrackState.Tracked]
            # iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
            iou_sim, iou_dist = iou_distance(tracks + dets, tracks + dets)

            scores = np.array([d.score for d in dets])

            # Run track aware NMS
            allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.det_init_thr)


            for idx, flag in enumerate(allow_indices):
                if flag:
                    dets[idx].initiate(self.frame_id, self.counter)
                    self.tracks.append(dets[idx])

        else:
            for det in dets:
                det.initiate(self.frame_id, self.counter)
                self.tracks.append(det)

    def update(self, dets, use_cmc, use_dlo, use_reid, use_tpa):
        self.frame_id += 1

        # 1. Convert detections -> Track objects
        dets = [Track(self.args, d) for d in dets]

        # boost confidence of detections
        if use_dlo:
            dets = apply_boost(dets, self.tracks, self.frame_id, True, True)
        
        # Split detections
        dets_high = [d for d in dets if d.score >= self.args.det_high_thr]
        dets_low  = [d for d in dets if self.args.det_low_thr <= d.score < self.args.det_high_thr]

        # 2. Split tracks
        tracked_lost = [t for t in self.tracks if t.state in [TrackState.Tracked, TrackState.Lost]]
        new_tracks   = [t for t in self.tracks if t.state == TrackState.New]

        if use_cmc:
            warp_matrix = self.cmc.get_warp_matrix()
            apply_cmc(tracked_lost, warp_matrix)
            apply_cmc(new_tracks, warp_matrix)


        # Predict
        [t.predict() for t in tracked_lost]
        [t.predict() for t in new_tracks]


        u_track = []
        dets_high_remain = []

        if not use_tpa:
            # =========================
            # --- STAGE 1: High-conf ---
            # =========================
            cost = build_cost_stage1(tracked_lost, dets_high, self.frame_id, use_reid)
            matches1, u_tracks1, u_dets1 = linear_assignment(cost, self.args.match_thr)

            for t, d in matches1:
                tracked_lost[t].update(self.frame_id, dets_high[d], update_feat=True)

            # =========================
            # --- STAGE 2: Low-conf ---
            # =========================
            remain_tracked = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Tracked]

            cost = build_cost_stage2(remain_tracked, dets_low, self.frame_id)
            matches2, u_tracks2, u_dets2 = linear_assignment(cost, 0.5)

            for t, d in matches2:
                remain_tracked[t].update(self.frame_id, dets_low[d], update_feat=False)

            # unmatched tracked -> lost
            for i in u_tracks2:
                remain_tracked[i].mark_lost()


            dets_high_remain = [dets_high[i] for i in u_dets1]
            u_track = u_tracks1


        else: 
            # =========================
            # STAGE12 (TPA: HIGH + LOW)
            # =========================
            dets_all = dets_high + dets_low

            cost = build_cost_stage12(tracked_lost, dets_high, dets_low, self.frame_id, use_reid, self.args.penalty_p)
            matches, u_tracks, u_dets = tpa_assignment(cost, self.args.match_thr, self.args.reduce_step)


            for t, d in matches:
                # tracked_lost[t].update(self.frame_id, dets_all[d], update_feat=True)
                tracked_lost[t].update(self.frame_id, dets_all[d], update_feat=True if d < len(dets_high) else False)

            # unmatched tracked → lost
            for i in u_tracks:
                if tracked_lost[i].state == TrackState.Tracked:
                    tracked_lost[i].mark_lost()


            dets_high_remain = [dets_all[i] for i in u_dets if i < len(dets_high)]
            u_track = u_tracks


        # =========================
        # --- STAGE 3: MERGED (Lost + New) ---
        # =========================
        remain_lost = [tracked_lost[i] for i in u_track if tracked_lost[i].state == TrackState.Lost]
        stage3_tracks = remain_lost + new_tracks

        cost = build_cost_stage3(stage3_tracks, dets_high_remain)
        matches3, u_tracks3, u_dets3 = linear_assignment(cost, self.args.match_thr)
        
        # cost = build_cost_stage12(stage3_tracks, dets_high_remain, [],self.frame_id, use_reid, self.args.penalty_p)
        # matches3, u_tracks3, u_dets3  = tpa_assignment(cost, self.args.match_thr, self.args.reduce_step)


        for t, d in matches3:
            update_feat = (stage3_tracks[t].state == TrackState.New)
            stage3_tracks[t].update(self.frame_id, dets_high_remain[d], update_feat)

        # unmatched new -> removed
        for t in stage3_tracks:
            if t.state == TrackState.New and t not in [stage3_tracks[i] for i, _ in matches3]:
                t.mark_removed()

        # lost quá lâu -> removed
        for t in self.tracks:
            if t.state == TrackState.Lost:
                if self.frame_id - t.end_frame_id > self.args.max_time_lost:
                    t.mark_removed()

        # =========================
        # --- INIT NEW TRACK ---
        # =========================
        new_dets = [dets_high_remain[i] for i in u_dets3]
        self.init_tracks(new_dets)

        # Clean removed
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
        # Update frame id
        self.frame_id += 1

        # Only maintain already tracked and new tracks, Drop all the new tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        # Camera motion compensation
        if self.cmc is not None: # change
            warp_matrix = self.cmc.get_warp_matrix()
            apply_cmc(self.tracks, warp_matrix)

        # Predict the current location with KF
        [t.predict() for t in self.tracks]

        # Change every track as lost tracks
        for t in self.tracks:
            t.mark_lost() 
            
        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return []
    