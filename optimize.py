import argparse
import time
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from preprocess import prepare_vio_inputs
from frontend import (
    extract_visual_measurements,
    project_kannala_brandt,
    so3_exp,
)
from settings import DEFAULT_CONFIG_PATH, load_run_config


# ==============================================================================
# 1. Lie Group SO(3) Helpers (Logarithm, Right Jacobian, Skew-Symmetric)
# ==============================================================================

def skew(v: np.ndarray) -> np.ndarray:
    """Returns 3x3 skew-symmetric matrix [v]_x."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) -> so(3) rotation vector."""
    cos_theta = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
    if np.pi - theta < 1e-5:
        # Near-pi fallback using eigenvector of eigenvalue +1
        eigvals, eigvecs = np.linalg.eig(R)
        axis = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
        axis /= np.linalg.norm(axis)
        return theta * axis
    factor = 0.5 * theta / np.sin(theta)
    return factor * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian J_r(phi) of SO(3)."""
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-7:
        return np.eye(3, dtype=np.float64) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    return (
        np.eye(3, dtype=np.float64)
        - ((1.0 - np.cos(theta)) / t2) * K
        + ((theta - np.sin(theta)) / (t2 * theta)) * (K @ K)
    )


def so3_inv_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian J_r^{-1}(phi) of SO(3)."""
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-7:
        return np.eye(3, dtype=np.float64) + 0.5 * K + (1.0 / 12.0) * (K @ K)
    t2 = theta * theta
    cot_half = np.cos(0.5 * theta) / np.sin(0.5 * theta)
    coeff = (1.0 / t2) - (cot_half / (2.0 * theta))
    return np.eye(3, dtype=np.float64) + 0.5 * K + coeff * (K @ K)


# ==============================================================================
# 2. On-Manifold 1000 Hz IMU Preintegrator (Forster et al. 2017)
# ==============================================================================

class PreintegratedIMUFactor:
    """
    Preintegrates 1000 Hz IMU measurements between frame i and frame j,
    maintaining preintegrated (dR, dv, dp), 9x9 covariance, and first-order
    bias Jacobians for O(1) bias updates during optimization.
    """

    def __init__(
        self,
        interval: dict,
        b_g_bar: np.ndarray,
        b_a_bar: np.ndarray,
        sigma_g: float,
        sigma_a: float,
    ):
        self.interval = interval
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.reintegrate(b_g_bar, b_a_bar)

    def reintegrate(self, b_g_bar: np.ndarray, b_a_bar: np.ndarray):
        self.b_g_bar = b_g_bar.copy()
        self.b_a_bar = b_a_bar.copy()

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

            omega = omega_raw - b_g_bar
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
                [Jr_step * dt, Z3],
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
            J_R_bg = dR_step.T @ J_R_bg - Jr_step * dt

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


# ==============================================================================
# 3. Vectorized Kannala-Brandt Projection & Exact Analytical Jacobian
# ==============================================================================

def project_kb_with_jacobian(pts_cam: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects N 3D points in camera frame (N, 3) -> 2D fisheye pixels (N, 2)
    and computes exact analytical Jacobians d(u, v)/d(X_cam) of shape (N, 2, 3).
    """
    N = len(pts_cam)
    uv = np.empty((N, 2), dtype=np.float64)
    J = np.zeros((N, 2, 3), dtype=np.float64)
    if N == 0:
        return uv, J

    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2]
    r2 = x * x + y * y
    r = np.sqrt(r2)
    d2 = np.maximum(r2 + z * z, 1e-12)
    theta = np.arctan2(r, z)

    k1, k2, k3, k4 = cam["D"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4

    rd = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)
    drd_dtheta = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8

    near_axis = r <= 1e-8
    safe_r = np.where(near_axis, 1.0, r)
    safe_r2 = np.where(near_axis, 1.0, r2)

    s = np.where(near_axis, 1.0 / np.maximum(z, 1e-6), rd / safe_r)
    uv[:, 0] = fx * (x * s) + cx
    uv[:, 1] = fy * (y * s) + cy

    # ds/dx, ds/dy, ds/dz
    common = (drd_dtheta * z / d2 - s) / safe_r2
    ds_dx = np.where(near_axis, 0.0, x * common)
    ds_dy = np.where(near_axis, 0.0, y * common)
    ds_dz = np.where(near_axis, -1.0 / np.maximum(z * z, 1e-6), -drd_dtheta / d2)

    J[:, 0, 0] = fx * (s + x * ds_dx)
    J[:, 0, 1] = fx * (x * ds_dy)
    J[:, 0, 2] = fx * (x * ds_dz)

    J[:, 1, 0] = fy * (y * ds_dx)
    J[:, 1, 1] = fy * (s + y * ds_dy)
    J[:, 1, 2] = fy * (y * ds_dz)

    return uv, J


# ==============================================================================
# 4. State Initialization & Observation Packing
# ==============================================================================

def initialize_vio_states(
    vio_data: dict,
    visual_meas: dict,
    preint_factors: list[PreintegratedIMUFactor],
    g_mag: float = 9.80665,
):
    """
    Initializes per-frame (R_wi, p_wi, v_w, bg, ba), 3D landmarks, and the
    gravity orientation R_g in World frame.
    """
    N = len(vio_data["t_cam1"])
    t_cam1 = np.asarray(vio_data["t_cam1"], dtype=np.float64)
    imu_intervals = vio_data["imu_intervals_cam1"]
    init_pnp = visual_meas["init_poses_world_imu"]
    b_gyro_0 = np.asarray(vio_data["b_gyro_0"], dtype=np.float64)

    R_wi = np.zeros((N, 3, 3), dtype=np.float64)
    p_wi = np.zeros((N, 3), dtype=np.float64)
    v_w = np.zeros((N, 3), dtype=np.float64)
    bg = np.tile(b_gyro_0, (N, 1))
    ba = np.zeros((N, 3), dtype=np.float64)

    # Estimate initial gravity direction in World frame from early stationary accel + R_wi[0]
    first_k = min(init_pnp.keys())
    R0 = init_pnp[first_k][:3, :3]
    accel_samples = []
    for k in range(min(15, len(imu_intervals))):
        accel_samples.append(imu_intervals[k]["accel"])
    mean_accel_I = np.mean(np.vstack(accel_samples), axis=0)
    # Specific force at rest: a_I = -R_IW * g_W  =>  g_W_dir = -R_WI * a_I / ||a_I||
    g_dir_w = -(R0 @ mean_accel_I)
    g_dir_w /= np.linalg.norm(g_dir_w)

    # Construct orthonormal R_g such that R_g @ [0, 0, -1] == g_dir_w
    z_col = -g_dir_w
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_col[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_col = np.cross(ref, z_col)
    x_col /= np.linalg.norm(x_col)
    y_col = np.cross(z_col, x_col)
    R_g = np.column_stack([x_col, y_col, z_col])
    g_world = R_g @ np.array([0.0, 0.0, -g_mag], dtype=np.float64)

    # Fill poses from PnP, falling back to IMU propagation if any frame was occluded
    for k in range(N):
        if k in init_pnp:
            R_wi[k] = init_pnp[k][:3, :3]
            p_wi[k] = init_pnp[k][:3, 3]
        elif k > 0:
            pf = preint_factors[k - 1]
            dt = pf.dt
            v_est = (p_wi[k - 1] - p_wi[max(0, k - 2)]) / max(t_cam1[k - 1] - t_cam1[max(0, k - 2)], 1e-3)
            R_wi[k] = R_wi[k - 1] @ pf.dR
            p_wi[k] = p_wi[k - 1] + v_est * dt + 0.5 * g_world * (dt * dt) + R_wi[k - 1] @ pf.dp
        else:
            R_wi[k] = R0

    # Central-difference initial velocities
    for k in range(N):
        if k == 0:
            v_w[k] = (p_wi[1] - p_wi[0]) / max(t_cam1[1] - t_cam1[0], 1e-4)
        elif k == N - 1:
            v_w[k] = (p_wi[N - 1] - p_wi[N - 2]) / max(t_cam1[N - 1] - t_cam1[N - 2], 1e-4)
        else:
            v_w[k] = (p_wi[k + 1] - p_wi[k - 1]) / max(t_cam1[k + 1] - t_cam1[k - 1], 1e-4)

    # Pack static 3D landmarks into contiguous array
    lm_dict = visual_meas["landmarks_world"]["cam1"]
    lm_ids = sorted(lm_dict.keys())
    tid_to_lm_idx = {tid: idx for idx, tid in enumerate(lm_ids)}
    landmarks = np.array([lm_dict[tid] for tid in lm_ids], dtype=np.float64) if lm_ids else np.zeros((0, 3))

    return R_wi, p_wi, v_w, bg, ba, landmarks, tid_to_lm_idx, R_g


def extract_frame_gyros(vio_data: dict) -> np.ndarray:
    """Extracts raw angular velocity sample closest to each master frame timestamp t_cam1[k]."""
    N = len(vio_data["t_cam1"])
    intervals = vio_data["imu_intervals_cam1"]
    omega_raw = np.zeros((N, 3), dtype=np.float64)
    for k in range(N):
        if k < len(intervals):
            omega_raw[k] = intervals[k]["gyro"][0]
        else:
            omega_raw[k] = intervals[-1]["gyro"][-1]
    return omega_raw


def pack_visual_observations(visual_meas: dict, tid_to_lm_idx: dict):
    """Flattens anchor tag corner observations and KLT track observations into vectorized arrays."""
    # Anchor tag corners (known 3D world points)
    anc_frames, anc_pts_w, anc_uv = [], [], []
    for obs in visual_meas["anchor_obs"]:
        f_idx = int(obs["frame_idx"])
        for pt_w, uv in zip(obs["corners_world"], obs["corners_uv"]):
            anc_frames.append(f_idx)
            anc_pts_w.append(pt_w)
            anc_uv.append(uv)

    anchor_packed = {
        "frame_idx": np.asarray(anc_frames, dtype=np.int64),
        "pts_world": np.asarray(anc_pts_w, dtype=np.float64).reshape(-1, 3),
        "uv_meas": np.asarray(anc_uv, dtype=np.float64).reshape(-1, 2),
    }

    # KLT tracks (optimizable 3D landmarks)
    klt_frames, klt_lm_idx, klt_uv = [], [], []
    klt_tracks = visual_meas["klt_tracks"]["cam1"]
    for tid, lm_idx in tid_to_lm_idx.items():
        for f_idx, uv, _ in klt_tracks[tid]:
            klt_frames.append(int(f_idx))
            klt_lm_idx.append(lm_idx)
            klt_uv.append(uv)

    klt_packed = {
        "frame_idx": np.asarray(klt_frames, dtype=np.int64),
        "lm_idx": np.asarray(klt_lm_idx, dtype=np.int64),
        "uv_meas": np.asarray(klt_uv, dtype=np.float64).reshape(-1, 2),
    }

    return anchor_packed, klt_packed


# ==============================================================================
# 5. Sparse Levenberg-Marquardt Batch VIO Solver
# ==============================================================================

def huber_irls_weights(residuals_2d: np.ndarray, delta_px: float) -> np.ndarray:
    """Computes IRLS sqrt-weights for 2D pixel reprojection residuals under Huber loss."""
    norms = np.linalg.norm(residuals_2d, axis=1)
    w = np.ones_like(norms)
    outliers = norms > delta_px
    w[outliers] = np.sqrt(delta_px / np.maximum(norms[outliers], 1e-8))
    return w


def evaluate_visual_residuals_and_jacobians(
    R_wi: np.ndarray,
    p_wi: np.ndarray,
    v_w: np.ndarray,
    bg: np.ndarray,
    omega_raw: np.ndarray,
    pts_world: np.ndarray,
    frame_idx: np.ndarray,
    uv_meas: np.ndarray,
    cam1: dict,
    dt_offset: float,
    sigma_px: float,
    huber_delta_px: float,
):
    """
    Vectorized evaluation of fisheye reprojection residuals and analytical
    Jacobians on the rear-camera timeline. The optional dt_offset term is zero
    for this rear-only pipeline.
    """
    K = len(frame_idx)
    if K == 0:
        return (
            np.empty(0),
            np.empty((0, 2)),
            np.empty((0, 2, 15)),
            np.empty((0, 2, 3)),
        )

    R_k = R_wi[frame_idx]          # (K, 3, 3)
    p_k = p_wi[frame_idx]          # (K, 3)
    v_k = v_w[frame_idx]           # (K, 3)
    bg_k = bg[frame_idx]           # (K, 3)
    w_k = omega_raw[frame_idx]     # (K, 3)

    R_k_T = np.transpose(R_k, (0, 2, 1))
    dX_w = pts_world - p_k
    X_I = np.einsum("kij,kj->ki", R_k_T, dX_w)

    # Optional camera time-offset compensation (dt_offset = 0 for rear-only).
    omega_unb = w_k - bg_k
    R_off = np.array([so3_exp(om * dt_offset) for om in omega_unb], dtype=np.float64)
    R_off_T = np.transpose(R_off, (0, 2, 1))
    v_I = np.einsum("kij,kj->ki", R_k_T, v_k)

    X_I_rot = np.einsum("kij,kj->ki", R_off_T, X_I)
    X_I_off = X_I_rot - dt_offset * v_I

    R_ci = cam1["T_cam_imu"][:3, :3]
    t_ci = cam1["T_cam_imu"][:3, 3]
    X_C = X_I_off @ R_ci.T + t_ci

    uv_pred, J_kb = project_kb_with_jacobian(X_C, cam1)  # (K, 2), (K, 2, 3)
    err_px = uv_pred - uv_meas  # (K, 2)

    # Huber IRLS weighting + pixel measurement noise scaling
    w_huber = huber_irls_weights(err_px, huber_delta_px)
    scale = (w_huber / sigma_px)[:, None]  # (K, 1)

    # Chain rule: d(uv)/d(X_I_off) = J_kb @ R_ci  -> shape (K, 2, 3)
    J_XI_off = np.einsum("kij,jl->kil", J_kb, R_ci)

    # Build skew-symmetric batches for X_I, v_I, X_I_rot
    def batch_skew(vecs: np.ndarray) -> np.ndarray:
        S = np.zeros((len(vecs), 3, 3), dtype=np.float64)
        S[:, 0, 1] = -vecs[:, 2]
        S[:, 0, 2] = vecs[:, 1]
        S[:, 1, 0] = vecs[:, 2]
        S[:, 1, 2] = -vecs[:, 0]
        S[:, 2, 0] = -vecs[:, 1]
        S[:, 2, 1] = vecs[:, 0]
        return S

    S_XI = batch_skew(X_I)
    S_vI = batch_skew(v_I)
    S_XI_rot = batch_skew(X_I_rot)

    J_state = np.zeros((K, 2, 15), dtype=np.float64)
    # d(X_I_off)/d(dphi_k) = R_off^T [X_I]_x - dt_offset [v_I]_x
    dX_dphi = np.einsum("kij,kjl->kil", R_off_T, S_XI) - dt_offset * S_vI
    J_state[:, :, 0:3] = np.einsum("kij,kjl->kil", J_XI_off, dX_dphi)

    # d(X_I_off)/d(dp_k) = -R_off^T R_k^T
    dX_dp = -np.einsum("kij,kjl->kil", R_off_T, R_k_T)
    J_state[:, :, 3:6] = np.einsum("kij,kjl->kil", J_XI_off, dX_dp)

    # d(X_I_off)/d(dv_k) = -dt_offset * R_k^T
    J_state[:, :, 6:9] = np.einsum("kij,kjl->kil", J_XI_off, -dt_offset * R_k_T)

    # d(X_I_off)/d(dbg_k) = -dt_offset * [R_off^T X_I]_x
    J_state[:, :, 9:12] = np.einsum("kij,kjl->kil", J_XI_off, -dt_offset * S_XI_rot)

    # d(uv)/d(X_world) = -d(uv)/d(dp_k)
    J_pt = -J_state[:, :, 3:6]

    res_weighted = (err_px * scale).ravel()
    J_state *= scale[:, :, None]
    J_pt *= scale[:, :, None]

    return res_weighted, err_px, J_state, J_pt


def run_batch_vio_optimization(
    vio_data: dict,
    visual_meas: dict,
    max_iters: int = 40,
    anchor_sigma_px: float = 0.05,
    klt_sigma_px: float = 0.40,
    huber_delta_px: float = 2.5,
    anchor_huber_delta_px: float = 10.0,
    init_prior_rot: float = 0.0,
    init_prior_trans: float = 0.0,
) -> dict:
    """
    Runs full-batch on-manifold Levenberg-Marquardt VIO bundle adjustment over all
    frames (656 IMU poses/velocities/biases), 3D KLT landmarks, and 2-DoF gravity tilt.

    Anchor corners use anchor_sigma_px/anchor_huber_delta_px; KLT uses
    klt_sigma_px/huber_delta_px.
    """
    t0 = time.perf_counter()
    N = len(vio_data["t_cam1"])
    cam1 = vio_data["cams"]["cam1"]
    # Rear camera is the only/master camera; observations are already on t_cam1.
    dt_offset = 0.0

    imu_cfg = vio_data.get("imu_cfg", {})
    g_mag = float(imu_cfg.get("g", 9.80665))
    sigma_g = float(imu_cfg.get("gyro_noise_density", 1.8e-4)) * np.sqrt(1000.0)
    sigma_a = float(imu_cfg.get("accel_noise_density", 2.0e-3)) * np.sqrt(1000.0)
    sigma_bg_rw = float(imu_cfg.get("gyro_random_walk", 2.0e-5))
    sigma_ba_rw = float(imu_cfg.get("accel_random_walk", 3.0e-4))

    # 1. Preintegrate all 1000 Hz IMU intervals
    b_gyro_0 = np.asarray(vio_data["b_gyro_0"], dtype=np.float64)
    preint_factors = [
        PreintegratedIMUFactor(iv, b_gyro_0, np.zeros(3), sigma_g=sigma_g, sigma_a=sigma_a)
        for iv in vio_data["imu_intervals_cam1"]
    ]

    # 2. Initialize states, landmarks, and gravity orientation
    R_wi, p_wi, v_w, bg, ba, landmarks, tid_to_lm_idx, R_g = initialize_vio_states(
        vio_data, visual_meas, preint_factors, g_mag=g_mag
    )
    M = len(landmarks)
    omega_raw = extract_frame_gyros(vio_data)
    anchor_packed, klt_packed = pack_visual_observations(visual_meas, tid_to_lm_idx)

    # Optional PnP branch-continuity prior (disabled by default). The final
    # rear-only comparison selects the branch with the anchor weight: a tight
    # anchor sigma (0.05 px) keeps the continuity-selected PnP branch, while
    # the looser default allows the high-motion mirror branch.
    init_pnp = visual_meas.get("init_poses_world_imu", {})
    init_frames = np.asarray(sorted(init_pnp.keys()), dtype=np.int64)
    if len(init_frames):
        init_R = np.stack([init_pnp[int(k)][:3, :3] for k in init_frames])
        init_p = np.stack([init_pnp[int(k)][:3, 3] for k in init_frames])
    else:
        init_R = np.zeros((0, 3, 3), dtype=np.float64)
        init_p = np.zeros((0, 3), dtype=np.float64)

    n_state_vars = 15 * N
    n_lm_vars = 3 * M
    grav_col_offset = n_state_vars + n_lm_vars
    total_vars = grav_col_offset + 2

    def build_system(R_cur, p_cur, v_cur, bg_cur, ba_cur, lm_cur, Rg_cur):
        g_world = Rg_cur @ np.array([0.0, 0.0, -g_mag], dtype=np.float64)
        G_basis = Rg_cur @ np.array(
            [[0.0, -g_mag], [g_mag, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )

        rows, cols, vals, res_list = [], [], [], []
        row_cursor = 0

        def add_block(r0: int, c0: int, blk: np.ndarray):
            nr, nc = blk.shape
            rr_idx, cc_idx = np.meshgrid(
                np.arange(r0, r0 + nr), np.arange(c0, c0 + nc), indexing="ij"
            )
            rows.append(rr_idx.ravel())
            cols.append(cc_idx.ravel())
            vals.append(blk.ravel())

        # --- Factor A: Initial Bias Prior at k = 0 (6 residuals) ---
        w_bg0 = 1.0 / 1e-3
        w_ba0 = 1.0 / 2e-2
        res_b0 = np.concatenate([
            (bg_cur[0] - b_gyro_0) * w_bg0,
            (ba_cur[0] - np.zeros(3)) * w_ba0,
        ])
        res_list.append(res_b0)
        add_block(row_cursor, 9, np.eye(3) * w_bg0)
        add_block(row_cursor + 3, 12, np.eye(3) * w_ba0)
        row_cursor += 6

        # --- Factor B: 1000 Hz IMU Preintegration + Bias Random Walk (15*(N-1) residuals) ---
        for k in range(N - 1):
            pf = preint_factors[k]
            res_imu, Ji, Jj, Jg = pf.evaluate(
                R_cur[k], p_cur[k], v_cur[k], bg_cur[k], ba_cur[k],
                R_cur[k + 1], p_cur[k + 1], v_cur[k + 1],
                g_world, G_basis,
            )
            W_imu = pf.sqrt_info
            res_list.append(W_imu @ res_imu)
            add_block(row_cursor, 15 * k, W_imu @ Ji)
            add_block(row_cursor, 15 * (k + 1), W_imu @ Jj)
            add_block(row_cursor, grav_col_offset, W_imu @ Jg)
            row_cursor += 9

            # Bias random walk
            dt = max(pf.dt, 1e-4)
            w_bg = 1.0 / (sigma_bg_rw * np.sqrt(dt))
            w_ba = 1.0 / (sigma_ba_rw * np.sqrt(dt))
            res_rw = np.concatenate([
                (bg_cur[k + 1] - bg_cur[k]) * w_bg,
                (ba_cur[k + 1] - ba_cur[k]) * w_ba,
            ])
            res_list.append(res_rw)
            add_block(row_cursor, 15 * k + 9, -np.eye(3) * w_bg)
            add_block(row_cursor, 15 * (k + 1) + 9, np.eye(3) * w_bg)
            add_block(row_cursor + 3, 15 * k + 12, -np.eye(3) * w_ba)
            add_block(row_cursor + 3, 15 * (k + 1) + 12, np.eye(3) * w_ba)
            row_cursor += 6

        # --- Factor B2: PnP branch-continuity prior ---
        if (init_prior_rot > 0.0 or init_prior_trans > 0.0) and len(init_frames):
            w_rot = 1.0 / init_prior_rot if init_prior_rot > 0.0 else 0.0
            w_trans = 1.0 / init_prior_trans if init_prior_trans > 0.0 else 0.0
            for idx, k in enumerate(init_frames):
                r_rot = so3_log(init_R[idx].T @ R_cur[k])
                r_trans = p_cur[k] - init_p[idx]
                res_list.append(np.concatenate([r_rot * w_rot, r_trans * w_trans]))
                if w_rot > 0.0:
                    add_block(
                        row_cursor, 15 * k,
                        w_rot * so3_inv_right_jacobian(r_rot),
                    )
                if w_trans > 0.0:
                    add_block(row_cursor + 3, 15 * k + 3, w_trans * np.eye(3))
                row_cursor += 6

        # --- Factor C: Stationary Anchor AprilTag Corners (cam1) ---
        # Anchor corners are precise (initial PnP residual ~0.26 px); keep them
        # tight and effectively use a plain L2 cost for them.
        res_anc, err_anc_px, J_anc_state, _ = evaluate_visual_residuals_and_jacobians(
            R_cur, p_cur, v_cur, bg_cur, omega_raw,
            anchor_packed["pts_world"],
            anchor_packed["frame_idx"],
            anchor_packed["uv_meas"],
            cam1, dt_offset,
            sigma_px=anchor_sigma_px,
            huber_delta_px=anchor_huber_delta_px,
        )
        K_anc = len(anchor_packed["frame_idx"])
        if K_anc > 0:
            res_list.append(res_anc)
            r_base = row_cursor + np.arange(2 * K_anc)[:, None]
            f_cols = (15 * anchor_packed["frame_idx"][:, None] + np.arange(15)[None, :])
            f_cols_2 = np.repeat(f_cols, 2, axis=0)
            rows.append(np.broadcast_to(r_base, f_cols_2.shape).ravel())
            cols.append(f_cols_2.ravel())
            vals.append(J_anc_state.reshape(2 * K_anc, 15).ravel())
            row_cursor += 2 * K_anc

        # --- Factor D: Rear-Cam KLT Landmarks Bundle Adjustment (cam1) ---
        K_klt = len(klt_packed["frame_idx"])
        err_klt_px = np.zeros((0, 2))
        if K_klt > 0 and M > 0:
            pts_klt_w = lm_cur[klt_packed["lm_idx"]]
            res_klt, err_klt_px, J_klt_state, J_klt_pt = evaluate_visual_residuals_and_jacobians(
                R_cur, p_cur, v_cur, bg_cur, omega_raw,
                pts_klt_w,
                klt_packed["frame_idx"],
                klt_packed["uv_meas"],
                cam1, dt_offset,
                sigma_px=klt_sigma_px,
                huber_delta_px=huber_delta_px,
            )
            res_list.append(res_klt)
            r_base = row_cursor + np.arange(2 * K_klt)[:, None]

            # State Jacobian entries
            f_cols = (15 * klt_packed["frame_idx"][:, None] + np.arange(15)[None, :])
            f_cols_2 = np.repeat(f_cols, 2, axis=0)
            rows.append(np.broadcast_to(r_base, f_cols_2.shape).ravel())
            cols.append(f_cols_2.ravel())
            vals.append(J_klt_state.reshape(2 * K_klt, 15).ravel())

            # Landmark Jacobian entries
            l_cols = (n_state_vars + 3 * klt_packed["lm_idx"][:, None] + np.arange(3)[None, :])
            l_cols_2 = np.repeat(l_cols, 2, axis=0)
            rows.append(np.broadcast_to(r_base, l_cols_2.shape).ravel())
            cols.append(l_cols_2.ravel())
            vals.append(J_klt_pt.reshape(2 * K_klt, 3).ravel())
            row_cursor += 2 * K_klt

        r_vec = np.concatenate(res_list)
        J_mat = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row_cursor, total_vars),
        ).tocsc()
        cost = 0.5 * float(np.dot(r_vec, r_vec))
        return cost, r_vec, J_mat, err_anc_px, err_klt_px

    def apply_update(R_cur, p_cur, v_cur, bg_cur, ba_cur, lm_cur, Rg_cur, dx: np.ndarray):
        R_new = R_cur.copy()
        p_new = p_cur.copy()
        v_new = v_cur.copy()
        bg_new = bg_cur.copy()
        ba_new = ba_cur.copy()

        for k in range(N):
            s = dx[15 * k : 15 * (k + 1)]
            R_new[k] = R_cur[k] @ so3_exp(s[0:3])
            p_new[k] += s[3:6]
            v_new[k] += s[6:9]
            bg_new[k] += s[9:12]
            ba_new[k] += s[12:15]

        lm_new = lm_cur + dx[n_state_vars:grav_col_offset].reshape(M, 3) if M > 0 else lm_cur.copy()
        dg = dx[grav_col_offset : grav_col_offset + 2]
        Rg_new = Rg_cur @ so3_exp(np.array([dg[0], dg[1], 0.0], dtype=np.float64))
        return R_new, p_new, v_new, bg_new, ba_new, lm_new, Rg_new

    # 3. Levenberg-Marquardt Optimization Loop
    cost, r_vec, J_mat, err_anc_0, err_klt_0 = build_system(
        R_wi, p_wi, v_w, bg, ba, landmarks, R_g
    )
    init_anc_rmse = float(np.sqrt(np.mean(np.sum(err_anc_0 ** 2, axis=1)))) if len(err_anc_0) else 0.0
    init_klt_rmse = float(np.sqrt(np.mean(np.sum(err_klt_0 ** 2, axis=1)))) if len(err_klt_0) else 0.0
    print(f"[Init] Cost: {cost:.1f} | Anchor RMSE: {init_anc_rmse:.2f} px | KLT RMSE: {init_klt_rmse:.2f} px")

    lam = 1e-3
    for it in range(max_iters):
        H = (J_mat.T @ J_mat).tocsc()
        g_vec = -(J_mat.T @ r_vec)
        diag_H = np.maximum(H.diagonal(), 1e-6)

        accepted = False
        for _ in range(8):
            H_damped = H.copy()
            H_damped.setdiag(H.diagonal() + lam * diag_H)
            dx = spsolve(H_damped, g_vec)
            if not np.all(np.isfinite(dx)):
                lam *= 10.0
                continue

            R_try, p_try, v_try, bg_try, ba_try, lm_try, Rg_try = apply_update(
                R_wi, p_wi, v_w, bg, ba, landmarks, R_g, dx
            )
            cost_try, r_try, J_try, err_anc_try, err_klt_try = build_system(
                R_try, p_try, v_try, bg_try, ba_try, lm_try, Rg_try
            )

            if cost_try < cost:
                step_norm = float(np.linalg.norm(dx))
                R_wi, p_wi, v_w, bg, ba, landmarks, R_g = (
                    R_try, p_try, v_try, bg_try, ba_try, lm_try, Rg_try
                )
                cost, r_vec, J_mat = cost_try, r_try, J_try
                err_anc_fin, err_klt_fin = err_anc_try, err_klt_try
                lam = max(lam * 0.3, 1e-7)
                accepted = True
                break
            else:
                lam *= 5.0

        if not accepted:
            print(f"[Iter {it + 1:02d}] Converged (no further descent).")
            break

        # Re-linearize preintegration if bias shifted significantly (> 0.015)
        reint_count = 0
        for k in range(N - 1):
            pf = preint_factors[k]
            if (
                np.linalg.norm(bg[k] - pf.b_g_bar) > 0.015
                or np.linalg.norm(ba[k] - pf.b_a_bar) > 0.05
            ):
                pf.reintegrate(bg[k], ba[k])
                reint_count += 1
        if reint_count > 0:
            cost, r_vec, J_mat, err_anc_fin, err_klt_fin = build_system(
                R_wi, p_wi, v_w, bg, ba, landmarks, R_g
            )

        anc_rmse = float(np.sqrt(np.mean(np.sum(err_anc_fin ** 2, axis=1)))) if len(err_anc_fin) else 0.0
        klt_rmse = float(np.sqrt(np.mean(np.sum(err_klt_fin ** 2, axis=1)))) if len(err_klt_fin) else 0.0
        print(
            f"[Iter {it + 1:02d}] Cost: {cost:.1f} | Anchor RMSE: {anc_rmse:.2f} px | "
            f"KLT RMSE: {klt_rmse:.2f} px | ||dx||: {step_norm:.2e}"
        )
        if step_norm < 1e-4:
            break

    elapsed = time.perf_counter() - t0
    print(f"Batch VIO optimization finished in {elapsed:.2f} s.")

    # Assemble 4x4 homogeneous transforms for IMU and rear camera (cam1)
    T_world_imu = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], N, axis=0)
    T_world_imu[:, :3, :3] = R_wi
    T_world_imu[:, :3, 3] = p_wi

    T_imu_cam1 = vio_data["cams"]["cam1"]["T_imu_cam"]
    T_world_cam1 = T_world_imu @ T_imu_cam1[None, :, :]

    gravity_world = R_g @ np.array([0.0, 0.0, -g_mag], dtype=np.float64)

    return {
        "t_cam1": np.asarray(vio_data["t_cam1"], dtype=np.float64),
        "T_world_imu": T_world_imu,
        "T_world_cam1": T_world_cam1,
        "v_world": v_w,
        "b_gyro": bg,
        "b_accel": ba,
        "landmarks_world": landmarks,
        "gravity_world": gravity_world,
    }


# ==============================================================================
# 6. Final Trajectory Rerun Logger
# ==============================================================================

def log_optimized_vio_to_rerun(opt_res: dict, out_rrd: str):
    """Logs only the final rear-camera trajectory to Rerun."""
    Path(out_rrd).parent.mkdir(parents=True, exist_ok=True)
    rr.init("vio_batch_optimized", spawn=False)
    rr.save(out_rrd)

    rear_pts = opt_res["T_world_cam1"][:, :3, 3].astype(np.float32)
    rr.log(
        "world/cam1_rear_trajectory",
        rr.LineStrips3D([rear_pts], radii=0.004),
        static=True,
    )


# ==============================================================================
# 7. CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    # Locate --config before building the full parser so YAML values can become
    # the defaults for every tunable argument. Explicit CLI flags still win.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_run_config(config_args.config)

    parser = argparse.ArgumentParser(description="Run full-batch IMU-Aided VIO optimization")
    parser.add_argument(
        "--config",
        default=config_args.config,
        help=f"VIO YAML configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--rrd", required=True, help="Path to raw base.rrd")
    parser.add_argument("--calib", default="config/cam-imu.yaml", help="Path to calib.yaml")
    parser.add_argument(
        "--anchor-id",
        type=int,
        default=cfg["anchor"]["id"],
        help="Stationary anchor AprilTag ID",
    )
    parser.add_argument(
        "--anchor-size",
        type=float,
        default=cfg["anchor"]["size_m"],
        help="Anchor tag side length (m)",
    )
    parser.add_argument(
        "--out-npz",
        default=None,
        help=(
            "Output trajectory .npz "
            "(default: logs/estimate/<rrd-stem>_trajectory.npz)"
        ),
    )
    parser.add_argument(
        "--viz",
        default=None,
        help=(
            "Output rear-trajectory Rerun file "
            "(default: logs/estimate/<rrd-stem>_vio.rrd)"
        ),
    )
    parser.add_argument(
        "--max-corners-per-cam",
        type=int,
        default=cfg["frontend"]["max_corners_per_cam"],
        help="Maximum tracked KLT corners per camera",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=cfg["frontend"]["min_track_len"],
        help="Minimum KLT track length in frames",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=cfg["optimizer"]["max_iters"],
        help="Maximum LM iterations",
    )
    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=cfg["optimizer"]["anchor_sigma_px"],
        help=(
            "Anchor corner noise sigma (px). 0.05 keeps the "
            "continuity-selected PnP branch; 0.30 selects the high-motion branch."
        ),
    )
    parser.add_argument(
        "--klt-sigma",
        type=float,
        default=cfg["optimizer"]["klt_sigma_px"],
        help="KLT reprojection noise sigma (px)",
    )
    parser.add_argument(
        "--anchor-huber",
        type=float,
        default=cfg["optimizer"]["anchor_huber_px"],
        help="Anchor Huber delta (px)",
    )
    parser.add_argument(
        "--klt-huber",
        type=float,
        default=cfg["optimizer"]["klt_huber_px"],
        help="KLT Huber delta (px)",
    )
    parser.add_argument(
        "--init-prior-rot",
        type=float,
        default=cfg["optimizer"]["init_prior_rot"],
        help="PnP branch-continuity rotation prior sigma (rad); 0 disables",
    )
    parser.add_argument(
        "--init-prior-trans",
        type=float,
        default=cfg["optimizer"]["init_prior_trans"],
        help="PnP branch-continuity translation prior sigma (m); 0 disables",
    )
    args = parser.parse_args()

    print(f"VIO config: {args.config}")

    # Derive distinctive output names from the input recording.
    rrd_stem = Path(args.rrd).stem
    if args.out_npz is None:
        args.out_npz = str(Path("logs/estimate") / f"{rrd_stem}_trajectory.npz")
    if args.viz is None:
        args.viz = str(Path("logs/estimate") / f"{rrd_stem}_vio.rrd")

    vio_data = prepare_vio_inputs(args.rrd, args.calib)

    anchor_tags = {
        args.anchor_id: {
            "size_m": args.anchor_size,
            "T_world_tag": np.eye(4, dtype=np.float64),
        }
    }

    visual_meas = extract_visual_measurements(
        vio_data,
        anchor_tags=anchor_tags,
        max_corners_per_cam=args.max_corners_per_cam,
        min_track_len=args.min_track_len,
    )

    opt_res = run_batch_vio_optimization(
        vio_data,
        visual_meas,
        max_iters=args.max_iters,
        anchor_sigma_px=args.anchor_sigma,
        klt_sigma_px=args.klt_sigma,
        huber_delta_px=args.klt_huber,
        anchor_huber_delta_px=args.anchor_huber,
        init_prior_rot=args.init_prior_rot,
        init_prior_trans=args.init_prior_trans,
    )

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **opt_res)
    print(f"Saved smoothed 6-DoF trajectory to {args.out_npz}")

    if args.viz:
        log_optimized_vio_to_rerun(opt_res, out_rrd=args.viz)
        print(f"Saved rear-camera trajectory to {args.viz}")
