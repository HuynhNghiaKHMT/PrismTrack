from trackers_new.cmc import *
from trackers_new.utils import *
from trackers_new.track import *
from trackers_new.association import build_cost_stage1, build_cost_stage2

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

    def str2bool(v):
        return v.lower() == "true"

    def init_tracks(self, dets):
        
        use_tai = self.args.tai # change
        use_tai = use_tai.lower() == "true" # change
        
        if use_tai:

            # Get alive tracks, iou_similarity, and scores
            # tracks = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.New]
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
    '''
    def update(self, dets, use_cmc=True, use_reid=True):
        self.frame_id += 1

        # 1. Phân loại detections
        dets = [Track(self.args, d) for d in dets]

        # D_high: Score > det_high_thr
        dets_high = [d for d in dets if d.score >= self.args.det_high_thr]
        
        # D_low: det_low_thr < Score < det_high_thr
        dets_low = [d for d in dets if self.args.det_low_thr <= d.score < self.args.det_high_thr]

        
        # 2. Phân loại Tracks hiện có
        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        if use_cmc:
            warp_matrix = self.cmc.get_warp_matrix()
            apply_cmc(tracked_lost, warp_matrix)
            apply_cmc(new, warp_matrix)

        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]


        # --- STAGE 1: High-confidence Association ---
        # Áp dụng cho: Track Confirmed (tracked + lost)
        matches1 = []
        u_tracks1 = list(range(len(tracked_lost)))
        u_dets1 = list(range(len(dets_high)))

        
        cost = build_cost_stage1(tracked_lost, dets_high, self.frame_id, use_reid)
        matches1, u_tracks1, u_dets1 = linear_assignment(cost, self.args.match_thr)

        for t, d in matches1:
            tracked_lost[t].update(self.frame_id, dets_high[d], update_feat=True)


        # --- STAGE 2: Low-confidence Association ---
        # Áp dụng cho: Track Confirmed (tracked + lost) còn sót lại, chưa match ở S1
        remain_tracked = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Tracked]

        matches2 = []
        u_tracks2 = list(range(len(remain_tracked)))
        u_dets2 = list(range(len(dets_low)))

        cost = build_cost_stage2(remain_tracked, dets_low)
        matches2, u_tracks2, u_dets2 = linear_assignment(cost, 0.5) 
        
        for t, d in matches2:
            remain_tracked[t].update(self.frame_id, dets_low[d], update_feat=False)


        # --- STAGE 2.5: Lost Track Association ---
        #  Áp dụng cho: Lost Tracks & High-confidence sau S1
        # remain_lost = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Lost]
        # dets_high_remain = [dets_high[i] for i in u_dets1]

        # matches2_ = []
        # u_tracks2_ = list(range(len(remain_lost)))
        # u_dets2_ = list(range(len(dets_high_remain)))

        # cost = build_cost_stage1(remain_lost, dets_high_remain, use_reid)

        # matches2_, u_tracks2_, u_dets2_ = linear_assignment(cost, self.args.match_thr)
        
        # for t, d in matches2_:
        #     remain_lost[t].update(self.frame_id, dets_high_remain[d], update_feat=True)



        # --- STAGE 3: New Track Association ---
        #  Áp dụng cho: Tracks Tentative & High-confidence sau S1
        # dets_high_remain = [dets_high_remain[i] for i in u_dets2_]
        dets_high_remain = [dets_high[i] for i in u_dets1]
        new_tracks= new

        matches3 = []
        u_tracks3 = list(range(len(new_tracks)))
        u_dets3 = list(range(len(dets_high_remain)))

        cost = build_cost_stage1(new_tracks, dets_high_remain,self.frame_id, use_reid)

        matches3, u_tracks3, u_dets3 = linear_assignment(cost, self.args.match_thr)
        
        for t, d in matches3:
            new_tracks[t].update(self.frame_id, dets_high_remain[d], update_feat=True)



        # Tracked Track không match ở Stage 2 -> Lost
         # unmatched tracked -> lost
        for i in u_tracks2:
            remain_tracked[i].mark_lost()


        # Track tentative không match ở Stage 3 -> Removed
        for i in u_tracks3:
            new_tracks[i].mark_removed()
            

        # Lost Track: age >  max_age -> Removed
        for t in self.tracks:
            if t.state == TrackState.Lost:
                if self.frame_id - t.end_frame_id > self.args.max_age:
                    t.mark_removed()

        # Khởi tạo D_high không match có, score > det_init_thr -> NewTrack
        new_dets = [dets_high_remain[i] for i in u_dets3]
        self.init_tracks(new_dets)


        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return [t for t in self.tracks if t.state == TrackState.Tracked]
    '''

    def update(self, dets, use_cmc=True, use_reid=True):
        self.frame_id += 1

        # 1. Convert detections -> Track objects
        dets = [Track(self.args, d) for d in dets]

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

        # =========================
        # --- STAGE 1: High-conf ---
        # =========================
        matches1 = []
        u_tracks1 = list(range(len(tracked_lost)))
        u_dets1   = list(range(len(dets_high)))

        cost = build_cost_stage1(tracked_lost, dets_high, self.frame_id, use_reid)
        matches1, u_tracks1, u_dets1 = linear_assignment(cost, self.args.match_thr)

        for t, d in matches1:
            tracked_lost[t].update(self.frame_id, dets_high[d], update_feat=True)

        # =========================
        # --- STAGE 2: Low-conf ---
        # =========================
        remain_tracked = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Tracked]

        matches2 = []
        u_tracks2 = list(range(len(remain_tracked)))
        u_dets2   = list(range(len(dets_low)))

        cost = build_cost_stage2(remain_tracked, dets_low)
        matches2, u_tracks2, u_dets2 = linear_assignment(cost, 0.5)

        for t, d in matches2:
            remain_tracked[t].update(self.frame_id, dets_low[d], update_feat=False)

        # unmatched tracked -> lost
        for i in u_tracks2:
            remain_tracked[i].mark_lost()

        # =========================
        # --- STAGE 3: MERGED (Lost + New) ---
        # =========================
        dets_high_remain = [dets_high[i] for i in u_dets1]

        remain_lost = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Lost]
        stage3_tracks = remain_lost + new_tracks

        matches3 = []
        u_tracks3 = list(range(len(stage3_tracks)))
        u_dets3   = list(range(len(dets_high_remain)))

        cost = build_cost_stage2(stage3_tracks, dets_high_remain)
        matches3, u_tracks3, u_dets3 = linear_assignment(cost, self.args.match_thr)

        for t, d in matches3:
            track = stage3_tracks[t]
            if track.state == TrackState.New:
                track.update(self.frame_id, dets_high_remain[d], update_feat=False)
            else:
                track.update(self.frame_id, dets_high_remain[d], update_feat=False)

        # =========================
        # --- HANDLE UNMATCHED ---
        # =========================

        # unmatched new -> removed
        for t in stage3_tracks:
            if t.state == TrackState.New and t not in [stage3_tracks[i] for i, _ in matches3]:
                t.mark_removed()

        # lost quá lâu -> removed
        for t in self.tracks:
            if t.state == TrackState.Lost:
                if self.frame_id - t.end_frame_id > self.args.max_age:
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
            t.mark_lost() # remove mark_lost because it in update_virtual
            # t.update_virtual(self.frame_id)  # add update_virtual ocsort
            
        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return []


