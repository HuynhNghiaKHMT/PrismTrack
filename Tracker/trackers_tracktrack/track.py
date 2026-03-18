import numpy as np
from trackers_tracktrack.utils import get_prev_box
from trackers_tracktrack.kalman_filter import KalmanFilter
from trackers_tracktrack.kalman_filter_add_oc import HybridKalmanFilter


def get_vel(b_1, b_2):
    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[0]**2 + deltas[1]**2) + 1e-5
    norm_lb = np.sqrt(deltas[0]**2 + deltas[3]**2) + 1e-5
    norm_rt = np.sqrt(deltas[2]**2 + deltas[1]**2) + 1e-5
    norm_rb = np.sqrt(deltas[2]**2 + deltas[3]**2) + 1e-5

    # Get velocities
    vel_lt = np.array([b_2[0] - b_1[0], b_2[1] - b_1[1]]) / norm_lt
    vel_lb = np.array([b_2[0] - b_1[0], b_2[3] - b_1[3]]) / norm_lb
    vel_rt = np.array([b_2[2] - b_1[2], b_2[1] - b_1[1]]) / norm_rt
    vel_rb = np.array([b_2[2] - b_1[2], b_2[3] - b_1[3]]) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=0)


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackCounter(object):
    track_count = 0

    def get_track_id(self):
        self.track_count += 1
        return self.track_count


class BaseTrack(object):
    track_id = 0
    end_frame_id = 0
    state = TrackState.New

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class Track(BaseTrack):
    def __init__(self, args, detection):
        # Initialize 1
        self.start_frame_id = None # add
        self.end_frame_id = 0 # bbd
        self.args = args
        self.box = detection[:4]  # x1y1x2y2
        self.score = detection[4]

        # Initialize 2
        self.delta_t = 3
        self.history = {}
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.velocity = np.zeros((4, 2))

        # Initialize 3
        self.alpha = 0.95
        # self.feat = detection[6:][np.newaxis, :].copy()
        self.feat = detection[6:][np.newaxis, :].copy() if args.reid else None # change

    # bbd
    # ========================
    def get_delta_tau(self, current_frame_id):
        # delta_tau chính là khoảng cách thời gian
        return max(1, current_frame_id - self.end_frame_id)
    # ========================
    
    def update_features(self, feat, score):
        # Update and normalize
        beta = self.alpha + (1 - self.alpha) * (1 - score)
        self.feat = beta * self.feat + (1 - beta) * feat
        self.feat /= np.linalg.norm(self.feat)

    def initiate(self, frame_id, counter):
        # Get new track id
        self.track_id = counter.get_track_id()
        self.start_frame_id = frame_id # add

        # Initiate Kalman filter
        if self.args.kf == "old":
            self.kalman_filter = KalmanFilter()
        else:
            self.kalman_filter = HybridKalmanFilter()
            
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())

        # Initiate history
        self.history[frame_id] = [self.box.copy(), self.score.copy(), self.mean.copy(),
                                  self.covariance.copy(), self.feat.copy()]

        # Initiate parameters
        self.end_frame_id = frame_id
        self.state = TrackState.New

    def predict(self):
        # Zero out the velocity of w and h when track is lost or new.
        if self.state != TrackState.Tracked and 'Dance' in self.args.data_path:
            self.mean[6] = 0
            self.mean[7] = 0

        # Predict
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, frame_id, detection):
        # Update Kalman filter & Feature
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,                                                   
            detection.cxcywh.copy(),
            frame_id, 
            detection.score)
          
        # update_features
        # # self.update_features(detection.feat.copy(), detection.score)
        if self.args.reid:
            self.update_features(detection.feat.copy(), detection.score)
        else:
            self.feat = None

        # Update Kalman filter's observed status and last observation
        self.kalman_filter.observed = True 
        self.kalman_filter.last_observation = detection.cxcywh.copy()
        
        # Update history
        self.history[frame_id] = [detection.box.copy(), detection.score, self.mean.copy(),
                                  self.covariance.copy(), self.feat.copy()]

        # Update velocity
        self.velocity = np.zeros((4, 2))
        # for d_t in range(1, self.delta_t + 1):
        #     prev_box = get_prev_box(self.history, frame_id, d_t).copy()
        
        # Fix logic get_prev_box for OC-SORT: 
        # ==================================================
        # Lấy danh sách các frame có dữ liệu thực (loại bỏ các frame None của OC-SORT)
        valid_history_keys = [k for k, v in self.history.items() if v is not None]
        
        for d_t in range(1, self.delta_t + 1):
            target_key = frame_id - d_t
            
            # Nếu target_key có dữ liệu thực, dùng nó. 
            # Nếu không (là None hoặc chưa có), dùng frame gần nhất có dữ liệu.
            if target_key in valid_history_keys:
                prev_box = self.history[target_key][0].copy()
            else:
                # Lấy frame gần nhất trong quá khứ mà valid
                latest_valid_key = max([k for k in valid_history_keys if k < frame_id])
                prev_box = self.history[latest_valid_key][0].copy()
        # ==================================================
            self.velocity += get_vel(prev_box, detection.x1y1x2y2) / d_t
        self.velocity /= self.delta_t

        # Update parameters
        self.box = detection.box.copy()
        self.score = detection.score
        self.end_frame_id = frame_id
        self.state = TrackState.Tracked if len(self.history.keys()) >= self.args.min_len else TrackState.New

    def update_virtual(self, frame_id):
        """
        Xử lý track khi bị mất dấu (Unmatched) theo cơ chế OC-SORT.
        Thay thế cho hàm mark_lost cũ.
        """
        self.state = TrackState.Lost
        
        # 1. Lưu None vào history để đánh dấu frame này không có quan sát thực tế
        # Việc này giúp Kalman Filter tính toán 'time_gap' chính xác khi re-match
        self.history[frame_id] = None 
        
        # 2. Đánh dấu vào KF để dừng Smoothing (vì không có z để neo vào)
        if hasattr(self.kalman_filter, 'observed'):
            self.kalman_filter.observed = False

    @property
    def cxcywh(self):
        # Get current position in bounding box format `(center x, center y, aspect ratio, height)`.
        if self.mean is None:
            cx = (self.box[0] + self.box[2]) / 2
            cy = (self.box[1] + self.box[3]) / 2
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            cx = self.mean[0]
            cy = self.mean[1]
            w = self.mean[2]
            h = self.mean[3]

        return np.array([cx, cy, w, h])

    @property
    def x1y1wh(self):
        # Get current position in bounding box format `(left top x, left top y, right bottom x, right bottom y)`.
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            w = self.mean[2]
            h = self.mean[3]

        return np.array([x1, y1, w, h])

    @property
    def x1y1x2y2(self):
        # Get current position in bounding box format `(left top x, left top y, right bottom x, right bottom y)`.
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            x2 = self.box[2]
            y2 = self.box[3]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            x2 = self.mean[0] + self.mean[2] / 2
            y2 = self.mean[1] + self.mean[3] / 2

        return np.array([x1, y1, x2, y2])


