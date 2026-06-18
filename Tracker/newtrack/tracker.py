from utils.cmc import *
from newtrack.utils import *
from newtrack.track import *
from newtrack.association import *
from newtrack.boost_confidence import *

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
        self.cmc = CMC(vid_name) if args.cmc == "true" else None 

    def update(self, dets, use_cmc, use_idcboost, use_reid, use_tpa, use_ttr):
        self.frame_id += 1

        # Convert detections -> Track objects
        dets = [Track(self.args, d) for d in dets]

        # Boost confidence of detections
        if use_idcboost:
            dets = IDCBoost(dets, self.tracks, self.args.boost_coef, self.args.det_thr, self.args.iou_limit,
                            self.frame_id, use_dlo = True, use_duo = True)
        
        # Split detections
        dets_high = [d for d in dets if d.score >= self.args.det_high_thr]
        dets_low  = [d for d in dets if self.args.det_low_thr <= d.score < self.args.det_high_thr]

        # Split tracks
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
            # ===================
            # STAGE 1: HIGH-CONF
            # ===================
            cost = build_cost_stage1(tracked_lost, dets_high, self.args.asso, self.frame_id, use_reid)
            matches1, u_tracks1, u_dets1 = linear_assignment(cost, self.args.match_thr)

            # matches1, u_tracks1, u_dets1 = iterative_assignment(
            #         cost, self.args.match_thr, self.args.reduce_step,
            #         tracked_lost, dets_high)

            for t, d in matches1:
                tracked_lost[t].update(self.frame_id, dets_high[d], update_feat=True)

            # ==================
            # STAGE 2: LOW-CONF
            # ==================
            remain_tracked = [tracked_lost[i] for i in u_tracks1 if tracked_lost[i].state == TrackState.Tracked]

            cost = build_cost_stage2(remain_tracked, dets_low, self.args.asso, self.frame_id)
            matches2, u_tracks2, u_dets2 = linear_assignment(cost, 0.50)

            # matches2, u_tracks2, u_dets2 = iterative_assignment(
            #         cost, 0.50, self.args.reduce_step,
            #         remain_tracked, dets_low)


            for t, d in matches2:
                remain_tracked[t].update(self.frame_id, dets_low[d], update_feat=False)

            # unmatched tracked -> lost
            for i in u_tracks2:
                remain_tracked[i].mark_lost()


            dets_high_remain = [dets_high[i] for i in u_dets1]
            u_track = u_tracks1


        else: 
            # ==============================
            # STAGE TPA (MERGE: HIGH + LOW)
            # ==============================
            dets_all = dets_high + dets_low

            cost = build_cost_stage12(
                tracked_lost, dets_high, dets_low, 
                self.args.asso, self.args.penalty_p, self.args.w_motion, 
                self.args.w_vel, self.args.w_conf, self.args.w_shape,
                self.frame_id, use_reid, self.args.dt)
            matches, u_tracks, u_dets = iterative_assignment(
                cost, self.args.match_thr, self.args.reduce_step, 
                tracked_lost, dets_all)
            
            # matches, u_tracks, u_dets = linear_assignment(cost, self.args.match_thr)
            
            for t, d in matches:
                update_feat = True if d < len(dets_high) else False
                tracked_lost[t].update(self.frame_id, dets_all[d], update_feat)

            # unmatched tracked → lost
            for i in u_tracks:
                if tracked_lost[i].state == TrackState.Tracked:
                    tracked_lost[i].mark_lost()


            dets_high_remain = [dets_all[i] for i in u_dets if i < len(dets_high)]
            u_track = u_tracks

        # ============================
        # STAGE 3: FINAL (LOST + NEW)
        # ============================
        final_tracks = []
        matches3 = []
        u_dets3 = []

        remain_lost = [tracked_lost[i] for i in u_track if tracked_lost[i].state == TrackState.Lost]

        if not use_tpa:
            final_tracks = remain_lost + new_tracks

            cost = build_cost_stage3(final_tracks, dets_high_remain, self.args.asso, self.frame_id)
            matches3, u_tracks3, u_dets3 = linear_assignment(cost, self.args.match_thr)

            # matches3, u_tracks3, u_dets3 = iterative_assignment(
            #     cost, self.args.match_thr, self.args.reduce_step, 
            #     final_tracks, dets_high_remain)
            
                
        else: 
            final_tracks = remain_lost + new_tracks

            cost = build_cost_stage12(
                final_tracks, dets_high_remain, [], 
                self.args.asso, 0.0, self.args.w_motion, 
                self.args.w_vel, self.args.w_conf, self.args.w_shape,
                self.frame_id, use_reid, self.args.dt)
            matches3, u_tracks3, u_dets3  = iterative_assignment(
                cost, self.args.match_thr, self.args.reduce_step, 
                final_tracks, dets_high_remain)
            
            # matches3, u_tracks3, u_dets3 = linear_assignment(cost, self.args.match_thr)

                
        for t, d in matches3:
            update_feat = (final_tracks[t].state == TrackState.New)
            final_tracks[t].update(self.frame_id, dets_high_remain[d], update_feat)


        # ===========================
        # TRACK LIFECYCLE MANAGEMENT
        # ===========================

        # unmatched new -> removed
        for t in final_tracks:
            if t.state == TrackState.New and t not in [final_tracks[i] for i, _ in matches3]:
                t.mark_removed()

        # unmatched track -> removed
        for t in self.tracks:
            if t.state == TrackState.Lost:
                if self.frame_id - t.end_frame_id > self.args.max_time_lost:
                    t.mark_removed()

        # init new track
        new_dets = [dets_high_remain[i] for i in u_dets3]
        for det in new_dets:
            det.initiate(self.frame_id, self.counter)
            self.tracks.append(det)

        # Clean removed
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        if not use_ttr:
            return [t for t in self.tracks if t.state == TrackState.Tracked]

        else:
            outputs = []
            for t in self.tracks:
                # CASE 1: vừa được confirm → flush toàn bộ history
                if t.is_confirmed and not t.has_flushed:
                    for (f_id, box, score) in t.pending_history:
                        outputs.append((f_id, t.track_id, box.copy(), score))
                    t.pending_history = []  # clear sau khi flush
                    t.has_flushed = True

                # CASE 2: đã confirm từ trước → output bình thường
                elif t.state == TrackState.Tracked:
                    outputs.append((self.frame_id, t.track_id, t.box.copy(), t.score))
            return outputs

    def update_without_detections(self, use_ttr):
        # Update frame id
        self.frame_id += 1

        # Only maintain already tracked and new tracks, Drop all the new tracks
        # self.tracks = [t for t in self.tracks if t.state != TrackState.New]

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

        if not use_ttr:
            return [t for t in self.tracks if t.state == TrackState.Tracked]

        else:
            outputs = []
            for t in self.tracks:
                # CASE 1: vừa được confirm → flush toàn bộ history
                if t.is_confirmed and not t.has_flushed:
                    for (f_id, box, score) in t.pending_history:
                        outputs.append((f_id, t.track_id, box.copy(), score))
                    t.pending_history = []  # clear sau khi flush
                    t.has_flushed = True

                # CASE 2: đã confirm từ trước → output bình thường
                elif t.state == TrackState.Tracked:
                    outputs.append((self.frame_id, t.track_id, t.box.copy(), t.score))
            return outputs

    