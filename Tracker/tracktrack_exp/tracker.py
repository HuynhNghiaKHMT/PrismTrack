from trackers.cmc import *
from tracktrack_exp.utils import *
from tracktrack_exp.track import *
from tracktrack_exp.association import *

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
            tracks = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.New]
            # iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
            iou_sim, iou_dist = iou_distance(tracks + dets, tracks + dets)

            scores = np.array([d.score for d in dets])

            # Run track aware NMS
            allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.init_thr)


            for idx, flag in enumerate(allow_indices):
                if flag:
                    dets[idx].initiate(self.frame_id, self.counter)
                    self.tracks.append(dets[idx])

        else:
            for det in dets:
                det.initiate(self.frame_id, self.counter)
                self.tracks.append(det)

    def update(self, dets, dets_95, use_cmc=True, use_reid=True):
        self.frame_id += 1

        dets_del = find_deleted_detections(dets, dets_95)
        dets = [Track(self.args, d) for d in dets]
        dets_del = [Track(self.args, d) for d in dets_del]

        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]

        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        if use_cmc:
            warp_matrix = self.cmc.get_warp_matrix()
            apply_cmc(tracked_lost, warp_matrix)
            apply_cmc(new, warp_matrix)

        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]

        # dets_all = dets_high + dets_low + dets_del_high
        # matches, u_tracks, u_dets = iterative_assignment(
        #     tracked_lost, dets_high, dets_low, dets_del_high,
        #     self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
        #     self.args.reduce_step, self.frame_id, d_t=3,use_reid=use_reid
        # )
        matches, u_tracks, u_dets, dets_all = run_association(
            tracked_lost, dets_high, dets_low, dets_del_high,
            self.args, self.frame_id, use_reid
        )

        for t, d in matches:
            tracked_lost[t].update(self.frame_id, dets_all[d])

        for t in u_tracks:
            # tracked_lost[t].mark_lost() # remove mark_lost because it in update_virtual
            tracked_lost[t].update_virtual(self.frame_id) # add update_virtual ocsort

        dets_high_left = [dets_all[i] for i in u_dets if i < len(dets_high)]

        # matches, u_tracks, u_dets = iterative_assignment(
        #     new, dets_high_left, [], [], self.args.match_thr,
        #     self.args.penalty_p, self.args.penalty_q,
        #     self.args.reduce_step, self.frame_id, d_t=3, use_reid=use_reid
        # )
        if len(new) > 0 and len(dets_high_left) > 0:
            cost = build_cost(new, dets_high_left, self.args, self.frame_id, use_reid)

            if self.args.assi == "global":
                matches, u_tracks, u_dets = global_assignment(cost, self.args.match_thr)
            else:
                matches, u_tracks, u_dets = local_assignment(cost, self.args.match_thr, self.args.reduce_step)
        else:
            matches, u_tracks, u_dets = [], list(range(len(new))), list(range(len(dets_high_left)))


        for t, d in matches:
            new[t].update(self.frame_id, dets_high_left[d])

        for t in u_tracks:
            new[t].mark_removed()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        # self.init_tracks([dets_high_left[udx] for udx in u_dets])
        self.init_tracks(dets_high_left) # change``


        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
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
        #     # t.mark_lost() # remove mark_lost because it in update_virtual
        #     t.update_virtual(self.frame_id)  # add update_virtual ocsort
            
        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return [t for t in self.tracks if t.state == TrackState.Tracked]


