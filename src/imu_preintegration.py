"""On-manifold 1000 Hz IMU preintegration factor..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np

from .lie_algebra import (
    skew,
    so3_exp,
    so3_inv_right_jacobian,
    so3_log,
    so3_right_jacobian,
)


class PreintegratedIMUFactor:
    """
    Preintegrates 1000 Hz IMU measurements between frame i and frame j,
    maintaining preintegrated (dR, dv, dp), 9x9 covariance, and first-order
    bias Jacobians for O(1) bias updates during optimization.

    Gyro measurements are corrected with an optional fixed scale/misalignment
    matrix C_g: omega = C_g @ (omega_raw - b_g).
    """

    def __init__(
        self,
        interval: dict,
        b_g_bar: np.ndarray,
        b_a_bar: np.ndarray,
        sigma_g: float,
        sigma_a: float,
        gyro_correction: np.ndarray | None = None,
    ):
        self.interval = interval
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        if gyro_correction is None:
            self.gyro_correction = np.eye(3, dtype=np.float64)
        else:
            self.gyro_correction = np.asarray(gyro_correction, dtype=np.float64).reshape(3, 3)
        self.reintegrate(b_g_bar, b_a_bar)

    def reintegrate(self, b_g_bar: np.ndarray, b_a_bar: np.ndarray):
        self.b_g_bar = b_g_bar.copy()
        self.b_a_bar = b_a_bar.copy()
        Cg = self.gyro_correction

        dts = self.interval["dt"]
        gyros = self.interval["gyro"]
        accels = self.interval["accel"]

        dR = np.eye(3, dtype=np.float64)
        dv = np.zeros(3, dtype=np.float64)
        dp = np.zeros(3, dtype=np.float64)
        dt_sum = 0.0

        # First-order bias Jacobians
        J_R_bg = np.zeros((3, 3), dtype=np.float64)
        J_v_bg = np.zeros((3, 3), dtype=np.float64)
        J_v_ba = np.zeros((3, 3), dtype=np.float64)
        J_p_bg = np.zeros((3, 3), dtype=np.float64)
        J_p_ba = np.zeros((3, 3), dtype=np.float64)

        # 9x9 preintegration noise covariance ordered as [dphi, dv, dp]
        cov = np.zeros((9, 9), dtype=np.float64)
        I3 = np.eye(3, dtype=np.float64)
        Z3 = np.zeros((3, 3), dtype=np.float64)

        has_midpoint_samples = len(gyros) == len(dts) + 1
        for k in range(len(dts)):
            dt = float(dts[k])
            if dt <= 0.0:
                continue

            if has_midpoint_samples:
                omega_raw = 0.5 * (gyros[k] + gyros[k + 1])
                accel_raw = 0.5 * (accels[k] + accels[k + 1])
            else:
                omega_raw = gyros[k]
                accel_raw = accels[k]

            omega = Cg @ (omega_raw - b_g_bar)
            accel = accel_raw - b_a_bar

            dR_step = so3_exp(omega * dt)
            Jr_step = so3_right_jacobian(omega * dt)
            a_skew = skew(accel)

            # Error-state transition A (9x9) and noise input B (9x6)
            A = np.block([
                [dR_step.T, Z3, Z3],
                [-dR @ a_skew * dt, I3, Z3],
                [-0.5 * (dR @ a_skew) * (dt * dt), I3 * dt, I3],
            ])
            B = np.block([
                [Jr_step @ Cg * dt, Z3],
                [Z3, dR * dt],
                [Z3, 0.5 * dR * (dt * dt)],
            ])
            Q_noise = np.diag([
                (self.sigma_g ** 2) / dt,
                (self.sigma_g ** 2) / dt,
                (self.sigma_g ** 2) / dt,
                (self.sigma_a ** 2) / dt,
                (self.sigma_a ** 2) / dt,
                (self.sigma_a ** 2) / dt,
            ])
            cov = A @ cov @ A.T + B @ Q_noise @ B.T

            # Propagate bias Jacobians (before updating dR, dv, dp)
            J_p_bg += J_v_bg * dt - 0.5 * (dR @ a_skew @ J_R_bg) * (dt * dt)
            J_p_ba += J_v_ba * dt - 0.5 * dR * (dt * dt)
            J_v_bg += -(dR @ a_skew @ J_R_bg) * dt
            J_v_ba += -dR * dt
            J_R_bg = dR_step.T @ J_R_bg - Jr_step @ Cg * dt

            # Propagate nominal preintegrated states
            dp += dv * dt + 0.5 * (dR @ accel) * (dt * dt)
            dv += (dR @ accel) * dt
            dR = dR @ dR_step
            dt_sum += dt

        self.dR = dR
        self.dv = dv
        self.dp = dp
        self.dt = dt_sum
        self.J_R_bg = J_R_bg
        self.J_v_bg = J_v_bg
        self.J_v_ba = J_v_ba
        self.J_p_bg = J_p_bg
        self.J_p_ba = J_p_ba

        # Square-root information matrix (Cholesky of info = inv(cov))
        cov_reg = cov + np.eye(9) * 1e-12
        info = np.linalg.inv(cov_reg)
        self.sqrt_info = np.linalg.cholesky(0.5 * (info + info.T)).T

    def evaluate(
        self,
        R_i: np.ndarray,
        p_i: np.ndarray,
        v_i: np.ndarray,
        bg_i: np.ndarray,
        ba_i: np.ndarray,
        R_j: np.ndarray,
        p_j: np.ndarray,
        v_j: np.ndarray,
        g_world: np.ndarray,
        G_basis: np.ndarray,
    ):
        """
        Computes unweighted 9D residual [r_dR, r_dv, r_dp] and analytical Jacobians
        w.r.t. state i (15D), state j (15D), and 2-DoF gravity tilt (2D).
        """
        dbg = bg_i - self.b_g_bar
        dba = ba_i - self.b_a_bar
        dt = self.dt

        phi_corr = self.J_R_bg @ dbg
        dR_corr = self.dR @ so3_exp(phi_corr)
        dv_corr = self.dv + self.J_v_bg @ dbg + self.J_v_ba @ dba
        dp_corr = self.dp + self.J_p_bg @ dbg + self.J_p_ba @ dba

        R_i_T = R_i.T
        r_dR = so3_log(dR_corr.T @ R_i_T @ R_j)
        dv_world = v_j - v_i - g_world * dt
        dp_world = p_j - p_i - v_i * dt - 0.5 * g_world * (dt * dt)

        r_dv = R_i_T @ dv_world - dv_corr
        r_dp = R_i_T @ dp_world - dp_corr
        res = np.concatenate([r_dR, r_dv, r_dp])

        # Analytical Jacobians
        Jr_inv = so3_inv_right_jacobian(r_dR)
        Jr_corr = so3_right_jacobian(phi_corr)
        exp_r_T = so3_exp(r_dR).T

        J_i = np.zeros((9, 15), dtype=np.float64)
        J_j = np.zeros((9, 15), dtype=np.float64)
        J_g = np.zeros((9, 2), dtype=np.float64)

        # r_dR Jacobians
        J_i[0:3, 0:3] = -Jr_inv @ (R_j.T @ R_i)
        J_i[0:3, 9:12] = -Jr_inv @ exp_r_T @ Jr_corr @ self.J_R_bg
        J_j[0:3, 0:3] = Jr_inv

        # r_dv Jacobians
        J_i[3:6, 0:3] = skew(R_i_T @ dv_world)
        J_i[3:6, 6:9] = -R_i_T
        J_i[3:6, 9:12] = -self.J_v_bg
        J_i[3:6, 12:15] = -self.J_v_ba
        J_j[3:6, 6:9] = R_i_T
        J_g[3:6, 0:2] = -R_i_T @ G_basis * dt

        # r_dp Jacobians
        J_i[6:9, 0:3] = skew(R_i_T @ dp_world)
        J_i[6:9, 3:6] = -R_i_T
        J_i[6:9, 6:9] = -R_i_T * dt
        J_i[6:9, 9:12] = -self.J_p_bg
        J_i[6:9, 12:15] = -self.J_p_ba
        J_j[6:9, 3:6] = R_i_T
        J_g[6:9, 0:2] = -0.5 * (R_i_T @ G_basis) * (dt * dt)

        return res, J_i, J_j, J_g
