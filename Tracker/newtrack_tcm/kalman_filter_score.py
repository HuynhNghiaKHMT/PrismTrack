import numpy as np
import scipy.linalg


class HybridKalmanFilterScore:

    def __init__(self):
        self.dim_x = 2   # [score, vscore]
        self.dim_z = 1   # measurement: score

        # State transition
        self.F = np.array([
            [1., 1.],
            [0., 1.]
        ])

        # Observation
        self.H = np.array([[1., 0.]])

        # Noise scale (rất quan trọng)
        self.std_pos = 1e-2
        self.std_vel = 1e-5

    # =========================
    # INIT
    # =========================
    def initiate(self, score):
        mean = np.array([score, 0.])  # vscore = 0

        std = [
            self.std_pos,
            self.std_vel
        ]

        covariance = np.diag(np.square(std))
        return mean, covariance

    # =========================
    # PROCESS NOISE Q
    # =========================
    def build_Q(self):
        std = [
            self.std_pos,
            self.std_vel
        ]
        return np.diag(np.square(std))

    # =========================
    # MEASUREMENT NOISE R
    # =========================
    def build_R(self, confidence, residual):

        std = [
            self.std_pos
        ]
        # confidence cao → noise nhỏ
        std = [(1 - confidence) * self.std_pos]
        std += 0.1 * residual

        return np.diag(np.square(std))

    # =========================
    # PREDICT
    # =========================
    def predict(self, mean, covariance):
        Q = self.build_Q()

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q

        return mean, covariance

    # =========================
    # PROJECT
    # =========================
    def project(self, mean, covariance, confidence=0.0, residual=0.0):
        R = self.build_R(confidence, residual)

        mean_proj = self.H @ mean
        cov_proj = self.H @ covariance @ self.H.T + R

        return mean_proj, cov_proj

    # =========================
    # UPDATE
    # =========================
    def update(self, mean, covariance, measurement, confidence=0.0, residual=0.0):

        mean_proj, cov_proj = self.project(mean, covariance, confidence, residual)

        chol, lower = scipy.linalg.cho_factor(cov_proj, lower=True)
        K = scipy.linalg.cho_solve(
            (chol, lower),
            (covariance @ self.H.T).T
        ).T

        innovation = measurement - mean_proj

        new_mean = mean + K @ innovation
        new_cov = covariance - K @ cov_proj @ K.T

        return new_mean, new_cov