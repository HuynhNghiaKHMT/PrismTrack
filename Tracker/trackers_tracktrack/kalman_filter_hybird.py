import numpy as np
import scipy.linalg


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class HybridKalmanFilter:

    def __init__(self):
        self.dim_x = 8
        self.dim_z = 4

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.

        self.H = np.eye(4, 8)

        self.std_pos = 1 / 20.
        self.std_vel = 1 / 160.

    def initiate(self, z):
        mean = np.zeros(8)
        mean[:4] = z

        w, h = z[2], z[3]

        std = [
            2*self.std_pos*w,
            2*self.std_pos*h,
            2*self.std_pos*w,
            2*self.std_pos*h,
            10*self.std_vel*w,
            10*self.std_vel*h,
            10*self.std_vel*w,
            10*self.std_vel*h,
        ]

        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def build_Q(self, mean):

        w, h = mean[2], mean[3]

        std = [
            self.std_pos*w,
            self.std_pos*h,
            self.std_pos*w,
            self.std_pos*h,
            self.std_vel*w,
            self.std_vel*h,
            self.std_vel*w,
            self.std_vel*h,
        ]

        return np.diag(np.square(std))

    def build_R(self, mean, confidence):

        w, h = mean[2], mean[3]

        std = [
            self.std_pos*w,
            self.std_pos*h,
            self.std_pos*w,
            self.std_pos*h,
        ]
        
        std = [(1 - confidence) * x for x in std]
        R = np.diag(np.square(std))

        return R

    def predict(self, mean, covariance):

        Q = self.build_Q(mean)

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q

        return mean, covariance
    
    def multi_predict(self, means, covariances):
        """
        means: (N, 8) - mảng các vector trạng thái
        covariances: (N, 8, 8) - mảng các ma trận hiệp phương sai
        """
        if len(means) == 0:
            return means, covariances

        # means[:, 2] là w, means[:, 3] là h
        std_pos_w = self.std_pos * means[:, 2]
        std_pos_h = self.std_pos * means[:, 3]
        std_vel_w = self.std_vel * means[:, 2]
        std_vel_h = self.std_vel * means[:, 3]

        # Xây dựng đường chéo cho Q (N, 8)
        Q_diag = np.stack([
            std_pos_w, std_pos_h, std_pos_w, std_pos_h,
            std_vel_w, std_vel_h, std_vel_w, std_vel_h
        ], axis=-1)
        Q = np.array([np.diag(np.square(q)) for q in Q_diag])

        # Dự báo Mean: x' = Fx
        # (N, 8, 8) @ (N, 8, 1) -> (N, 8)
        means = np.matmul(self.F, means[..., np.newaxis]).squeeze(-1)

        # Dự báo Covariance: P' = FPF^T + Q
        # np.matmul hỗ trợ broadcasting tự động cho chiều N
        covariances = np.matmul(np.matmul(self.F, covariances), self.F.T) + Q

        return means, covariances
    

    def project(self, mean, covariance, confidence = .0):

        R = self.build_R(mean, confidence)

        mean_proj = self.H @ mean
        cov_proj = self.H @ covariance @ self.H.T + R

        return mean_proj, cov_proj

    def update(self, mean, covariance, z, confidence = .0):

        mean_proj, cov_proj = self.project(mean, covariance, confidence)

        chol, lower = scipy.linalg.cho_factor(cov_proj, lower=True)
        K = scipy.linalg.cho_solve(
            (chol, lower),
            (covariance @ self.H.T).T
        ).T

        innovation = z - mean_proj

        new_mean = mean + K @ innovation
        new_cov = covariance - K @ cov_proj @ K.T

        return new_mean, new_cov


    
    