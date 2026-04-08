from trackers.cmc import *
from newtrack_oru.utils import *
from newtrack_oru.track import *
from newtrack_oru.association import *
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
        self.cmc = CMC(vid_name) if args.cmc == "true" else None # change

        self.use_last_obs = True

    def init_tracks(self, dets):
    
        for det in dets:
            det.initiate(self.frame_id, self.counter)
            self.tracks.append(det)

    def update(self, dets, use_cmc, use_dlo, use_reid, use_tpa):
        self.frame_id += 1

        # 1. Convert detections -> Track objects
        dets = [Track(self.args, d) for d in dets]

        # boost confidence of detections
        if not use_dlo:
            dets = apply_boost(dets, self.tracks, self.frame_id, True, True)

        # Split detections
        dets_high = [d for d in dets if d.score >= self.args.det_high_thr]
        dets_low  = [d for d in dets if self.args.det_low_thr <= d.score < self.args.det_high_thr]

        # 2. Split tracks
        tracked_lost = [t for t in self.tracks if t.state in [TrackState.Tracked, TrackState.Lost]]
        new_tracks   = [t for t in self.tracks if t.state == TrackState.New]

        if not use_cmc:
            warp_matrix = self.cmc.get_warp_matrix()
            apply_cmc(tracked_lost, warp_matrix)
            apply_cmc(new_tracks, warp_matrix)


        # Predict
        [t.predict() for t in tracked_lost]
        [t.predict() for t in new_tracks]


        u_track = []
        dets_high_remain = []

       
        # STAGE 1: High-conf 
        # print (f"[STAGE 1] {len(tracked_lost)} tracked/lost vs {len(dets_high)} high-conf dets")
        cost = build_cost_stage1(tracked_lost, dets_high, self.frame_id, use_reid)
        matches1, u_tracks1, u_dets1 = linear_assignment(cost, self.args.match_thr)

        for t, d in matches1:
            tracked_lost[t].update(self.frame_id, dets_high[d], update_feat=True)

        
        #  STAGE 2: Low-conf 
        # print (f"[STAGE 2] {len(u_tracks1)} tracked/lost vs {len(dets_low)} low-conf dets")
        remain_tracked = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Tracked]

        cost = build_cost_stage2(remain_tracked, dets_low, self.frame_id)
        matches2, u_tracks2, u_dets2 = linear_assignment(cost, 0.5)

        for t, d in matches2:
            remain_tracked[t].update(self.frame_id, dets_low[d])

        
        # unmatched tracked -> lost
        for i in u_tracks2:
            remain_tracked[i].mark_lost()


        # STAGE 3: Lost TRACK

        dets_high_remain = [dets_high[i] for i in u_dets1]
        remain_lost = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Lost]
        
        # ---- KEY: dùng last_observation cho LOST khi build cost ----
        boxes_tracks = []
        for t in remain_lost:
            if t.state == TrackState.Lost and self.use_last_obs and t.last_observation is not None:
                boxes_tracks.append(t.last_observation)
            else:
                boxes_tracks.append(t.x1y1x2y2)


        # build cost custom (truyền box override)
        cost = build_cost_stage3_for_obs(remain_lost, dets_high_remain, track_boxes=boxes_tracks)
        matches3, u_tracks3, u_dets3 = linear_assignment(cost, 0.5)

        for t, d in matches3:
            remain_lost[t].update(self.frame_id, dets_high_remain[d], update_feat=True)

        # ===== STAGE 4: NEW tracks =====
        unmatched_dets = [dets_high_remain[i] for i in u_dets3]

        cost_new = build_cost_stage3(new_tracks, unmatched_dets)
        matches_new, u_tracks_new, u_dets_new = linear_assignment(cost_new, 0.5)

        for t, d in matches_new:
            new_tracks[t].update(self.frame_id, unmatched_dets[d], update_feat=True)

        # MANAGE UNMATCHED

        # Stage1 unmatched
        for i in u_tracks1:
            t = tracked_lost[i]
            if t.state == TrackState.Lost:
                t.update(self.frame_id, None)

        # Stage2 unmatched
        for i in u_tracks2:
            remain_tracked[i].update(self.frame_id, None)

        # matched_track_idx = set([t for t, _ in matches3])

        # # unmatched new -> removed
        # for i, t in enumerate(stage3_tracks):
        #     if t.state == TrackState.New and i not in matched_track_idx:
        #         t.mark_removed()

        for t in new_tracks:
            if t.state == TrackState.New and t not in [new_tracks[i] for i, _ in matches_new]:
                t.mark_removed()

        # lost quá lâu -> removed
        for t in self.tracks:
            if t.state == TrackState.Lost:
                if self.frame_id - t.end_frame_id > self.args.max_age:
                    t.mark_removed()


        # init new tracks
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
        # for t in self.tracks:
        #     t.mark_lost() 

        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return []