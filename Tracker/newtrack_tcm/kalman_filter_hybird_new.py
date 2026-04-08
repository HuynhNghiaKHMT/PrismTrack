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

class HybridKalmanFilterNew:

    def __init__(self):
        self.dim_x = 10
        self.dim_z = 5

        self.F = np.eye(10)
        for i in range(5):
            self.F[i, i + 5] = 1.

        self.H = np.eye(5, 10)

        self.std_pos = 1 / 20.
        self.std_vel = 1 / 160.

    def initiate(self, z, score):
        mean = np.zeros(10)
        mean[:4] = z
        mean[4] = score 

        w, h = z[2], z[3]

        std = [
            2*self.std_pos*w,
            2*self.std_pos*h,
            2*self.std_pos*w,
            2*self.std_pos*h,
            0.1,
            10*self.std_vel*w,
            10*self.std_vel*h,
            10*self.std_vel*w,
            10*self.std_vel*h,
            0.1,
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
            0.05,
            self.std_vel*w,
            self.std_vel*h,
            self.std_vel*w,
            self.std_vel*h,
            0.01
        ]

        return np.diag(np.square(std))

    def build_R(self, mean, confidence):

        w, h = mean[2], mean[3]

        std = [
            self.std_pos*w,
            self.std_pos*h,
            self.std_pos*w,
            self.std_pos*h,
            0.1 
        ]

        # chỉ scale bbox
        for i in range(4):
            confidence = np.clip(confidence, 0.01, 0.99)
            std[i] *= (1 - confidence)

        R = np.diag(np.square(std))

        return R

    def predict(self, mean, covariance):

        Q = self.build_Q(mean)

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q

        return mean, covariance

    def project(self, mean, covariance, confidence = .0):

        R = self.build_R(mean, confidence)

        mean_proj = self.H @ mean
        cov_proj = self.H @ covariance @ self.H.T + R

        return mean_proj, cov_proj

    def update(self, mean, covariance, z, score):

        measurement = np.concatenate([z, [score]])

        mean_proj, cov_proj = self.project(mean, covariance, score)

        chol, lower = scipy.linalg.cho_factor(cov_proj, lower=True)
        K = scipy.linalg.cho_solve(
            (chol, lower),
            (covariance @ self.H.T).T
        ).T

        innovation = measurement - mean_proj

        new_mean = mean + K @ innovation
        new_cov = covariance - K @ cov_proj @ K.T

        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        # Project to measurement space
        mean, covariance = self.project(mean, covariance)

        # mean: [x, y, w, h]
        # measurements: Nx4 (x, y, w, h)

        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]

        # Cholesky decomposition
        cholesky_factor = np.linalg.cholesky(covariance)

        # Residual
        d = measurements - mean

        # Solve
        z = scipy.linalg.solve_triangular(
            cholesky_factor,
            d.T,
            lower=True,
            check_finite=False,
            overwrite_b=True
        )

        squared_maha = np.sum(z * z, axis=0)
        return squared_maha