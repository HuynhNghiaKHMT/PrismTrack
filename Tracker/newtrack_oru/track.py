import numpy as np
from newtrack_oru.utils import get_prev_box, get_vel
from tracktrack_exp.kalman_filter import KalmanFilter
from newtrack_oru.kalman_filter_hybird import HybridKalmanFilter
from copy import deepcopy
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
        self.args = args
        self.box = detection[:4]  # x1y1x2y2
        self.score = detection[4]
        self.new_score = detection[4]

        # Initialize 2
        self.delta_t = 3
        self.history = {}
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.velocity = np.zeros((4, 2))

        # Initialize 3
        self.alpha = 0.95
        self.feat = detection[6:][np.newaxis, :].copy() if args.reid else None # change

        # Initialize 3
        self.last_observation = None
        self.observed = False
        self.attr_saved = None
        self.history_obs = []

    def get_delta_tau(self, current_frame_id):
        # delta_tau chính là khoảng cách thời gian
        return max(1, current_frame_id - self.end_frame_id)
    
    def freeze(self):
        
        self.attr_saved = deepcopy((self.mean.copy(), self.covariance.copy()))

    def unfreeze(self, current_box_z):
        if self.attr_saved is None:
            return

        # Khôi phục trạng thái cũ
        old_mean, old_cov = deepcopy(self.attr_saved)
        
        valid_obs = [z for z in self.history_obs if z is not None]
        if len(valid_obs) < 1: return
        last_z = valid_obs[-1]

        # Tính dịch chuyển thực tế từ quan sát (Observed Displacement)
        dx = (current_box_z - last_z)

        # Thay vì cộng thẳng vào self.mean hiện tại (đã qua predict), 
        # ta thiết lập lại mean dựa trên mỏ neo cũ + dịch chuyển thực tế
        # Điều này loại bỏ hoàn toàn sai số tích lũy của predict() trong thời gian LOST
        new_mean = old_mean.copy()
        new_mean[:4] += dx
        
        # Giữ nguyên vận tốc từ dự báo KF hoặc reset nhẹ để tránh vọt
        self.mean = new_mean
        self.covariance = old_cov # Dùng lại hiệp phương sai cũ vì nó "tin cậy" hơn cái đã bị predict quá nhiều lần

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
        self.kalman_filter = HybridKalmanFilter()
            
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())

        # Initiate history
        self.history[frame_id] = [self.box.copy(), self.score.copy(), self.mean.copy(),
                                  self.covariance.copy(), self.feat.copy()]

        # Initiate parameters
        self.end_frame_id = frame_id
        self.hits = 1
        self.state = TrackState.New

    def predict(self):
        # Zero out the velocity of w and h when track is lost or new.
        if self.state != TrackState.Tracked and 'Dance' in self.args.data_path:
            self.mean[6] = 0
            self.mean[7] = 0

        # Predict
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, frame_id, detection, update_feat = False):
        # ======================
        # OBS UPDATE
        # ======================
        # CASE 1: MISS (Không có detection)
        if detection is None:
            # Nếu frame này không có detection, ta KHÔNG cập nhật mean/covariance bằng KF update
            # Nhưng quan trọng: KHÔNG được update last_observation
            self.observed = False
            self.history_obs.append(None)
            return
        
        # Lấy quan sát hiện tại
        z_current = detection.cxcywh.copy()

        # Logic Unfreeze: Phải thực hiện TRƯỚC KHI thực hiện KF.update
        if not self.observed and self.attr_saved is not None:
            # Lấy mean/cov từ lúc bắt đầu bị LOST
            self.mean, self.covariance = self.attr_saved
            
            valid_obs = [z for z in self.history_obs if z is not None]
            if len(valid_obs) >= 1:
                last_z = valid_obs[-1]
                # Chỉ bù đắp vị trí (center x, y), không nên bù đắp w, h để tránh méo box
                dx = (z_current[:2] - last_z[:2])
                self.mean[:2] += dx

        self.observed = True
        self.history_obs.append(z_current)

        # CẬP NHẬT QUAN TRỌNG: 
        # Luôn thực hiện predict một bước nhỏ trước khi update để đồng bộ thời gian
        # (Hoặc đảm bảo hàm update của bạn nhận diện đúng khoảng cách frame)
        
        self.mean, self.covariance = self.kalman_filter.update( 
            self.mean, self.covariance, z_current, detection.score
        )

        # Lưu lại detection THỰC TẾ để làm mỏ neo cho lần Lost tiếp theo
        self.last_observation = detection.box.copy() 
        self.freeze()

        # ======================
        # NORMAL UPDATE
        # ======================
        # Update Kalman filter & Feature
        self.mean, self.covariance = self.kalman_filter.update( 
            self.mean, self.covariance,                                    
            detection.cxcywh.copy(),
            detection.score
        )

        self._post_update(frame_id, detection, update_feat)

    def _post_update(self, frame_id, detection, update_feat):
        # Cập nhật last_observation bằng tọa độ x1y1x2y2 của detection THỰC TẾ
        self.last_observation = detection.box.copy() # x1y1x2y2
          
        # update_features
        # Stage 1 mới cập nhật feature, Stage 2 thì không
        if update_feat and self.args.reid:
            self.update_features(detection.feat, detection.score)
        
        # Update history
        self.history[frame_id] = [detection.box.copy(), detection.score, self.mean.copy(),
                                  self.covariance.copy(), self.feat.copy() if self.feat is not None else None]

        # Update velocity
        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, frame_id, d_t).copy()
            self.velocity += get_vel(prev_box, detection.x1y1x2y2) / d_t
        self.velocity /= self.delta_t

        # Update parameters
        self.box = detection.box.copy()
        self.score = detection.score
        self.new_score = detection.score
        self.end_frame_id = frame_id
        self.hits += 1

        # self.state = TrackState.Tracked if len(self.history.keys()) >= self.args.min_hits else TrackState.New
        self.state = TrackState.Tracked if self.hits >= self.args.min_hits else TrackState.New

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

    @property
    def last_box(self):
        if self.last_observation is not None:
            return self.last_observation
        return self.x1y1x2y2
    
    @property
    def x1y1x2y2_last(self):
        if self.last_observation is not None:
            return self.last_observation
        return self.x1y1x2y2
    


 # CASE 1: MISS (Không có detection)
"""
def unfreeze(self, current_box_z):
        if self.attr_saved is None:
            return

        # restore state trước khi mất track
        self.mean, self.covariance = self.attr_saved

        # Lấy quan sát cuối cùng trước khi mất track
        valid_obs = [z for z in self.history_obs if z is not None]
        if len(valid_obs) < 1:
            return

        last_z = valid_obs[-1]

        # Tính toán vận tốc trung bình dựa trên sự thay đổi vị trí 
        # từ quan sát cuối đến quan sát hiện tại
        # Đây là core logic của Hybrid: bù đắp sai số dự báo bằng dịch chuyển thực tế
        dx = (current_box_z - last_z)

        # Cập nhật lại mean (vị trí) dựa trên dịch chuyển thực tế
        self.mean[:4] += dx
        

update()
if detection is None:
    if self.observed:
        self.freeze() # Lưu lại trạng thái tốt cuối cùng
    self.observed = False
    self.history_obs.append(None)
    return

# Lấy quan sát hiện tại
z_current = detection.cxcywh.copy()

# CASE 2: RE-APPEAR (Tìm thấy lại sau khi mất)
if not self.observed:
    self.unfreeze(z_current) # Điều chỉnh lại mean dựa trên dịch chuyển thực tế

self.observed = True
self.history_obs.append(z_current)
"""