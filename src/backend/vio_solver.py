"""Sparse Levenberg-Marquardt batch VIO solver..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ..core.camera_model import project_kb_with_jacobian
from ..core.lie_algebra import so3_exp, so3_inv_right_jacobian, so3_log
from .imu_preintegration import PreintegratedIMUFactor
from .vio_initialization import (
    extract_frame_gyros,
    initialize_vio_states,
    pack_visual_observations,
)


def huber_irls_weights(residuals_2d: np.ndarray, delta_px) -> np.ndarray:
    """Computes IRLS sqrt-weights for 2D pixel reprojection residuals under Huber loss."""
    norms = np.linalg.norm(residuals_2d, axis=1)
    delta = np.asarray(delta_px, dtype=np.float64)
    w = np.ones_like(norms)
    outliers = norms > delta
    if np.any(outliers):
        delta_out = delta[outliers] if delta.ndim > 0 else delta
        w[outliers] = np.sqrt(delta_out / np.maximum(norms[outliers], 1e-8))
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
    gyro_correction: np.ndarray | None = None,
    omega_rate_scale: float = 0.0,
):
    """
    Vectorized evaluation of fisheye reprojection residuals and analytical
    Jacobians on the rear-camera timeline.

    ``dt_offset`` is the current camera-minus-IMU timing estimate. Each row of
    the image applies a rolling-shutter correction using the sensor readout
    stored in ``cam1['rolling_shutter_readout_s']`` (seconds):
        dt_eff = dt_offset + (v - h/2)/h * readout_s
    so the frame timestamp is treated as the center-row exposure time.

    Gyro measurements are corrected with optional fixed scale/misalignment
    C_g: omega_unb = C_g @ (omega_raw - b_g). When omega_rate_scale > 0,
    per-observation sigma and Huber threshold are inflated by
    (1 + omega_rate_scale * ||omega_unb||) to downweight motion-blurred /
    high-angular-rate measurements.

    Returns (residual, pixel_error, J_state, J_point, J_time_offset).
    """
    K = len(frame_idx)
    if K == 0:
        return (
            np.empty(0),
            np.empty((0, 2)),
            np.empty((0, 2, 15)),
            np.empty((0, 2, 3)),
            np.empty((0, 2)),
        )

    R_k = R_wi[frame_idx]          # (K, 3, 3)
    p_k = p_wi[frame_idx]          # (K, 3)
    v_k = v_w[frame_idx]           # (K, 3)
    bg_k = bg[frame_idx]           # (K, 3)
    w_k = omega_raw[frame_idx]     # (K, 3)

    R_k_T = np.transpose(R_k, (0, 2, 1))
    dX_w = pts_world - p_k
    X_I = np.einsum("kij,kj->ki", R_k_T, dX_w)

    # Bias-corrected (and optionally scale/misalignment-corrected) body rates.
    omega_unb_raw = w_k - bg_k
    if gyro_correction is None:
        omega_unb = omega_unb_raw
        Cg = np.eye(3, dtype=np.float64)
    else:
        Cg = np.asarray(gyro_correction, dtype=np.float64).reshape(3, 3)
        omega_unb = (Cg @ omega_unb_raw.T).T

    # Camera/IMU time offset plus per-row rolling-shutter exposure shift.
    # Frame timestamps are assumed to represent center-row exposure.
    h_px = float(cam1["resolution"][1])
    readout = float(cam1.get("rolling_shutter_readout_s") or 0.0)
    row_fraction = (uv_meas[:, 1] - 0.5 * h_px) / h_px
    dt_eff = float(dt_offset) + row_fraction * max(readout, 0.0)  # (K,)

    R_off = np.array(
        [so3_exp(om * dt) for om, dt in zip(omega_unb, dt_eff)], dtype=np.float64
    )
    R_off_T = np.transpose(R_off, (0, 2, 1))
    v_I = np.einsum("kij,kj->ki", R_k_T, v_k)

    X_I_rot = np.einsum("kij,kj->ki", R_off_T, X_I)
    X_I_off = X_I_rot - dt_eff[:, None] * v_I

    R_ci = cam1["T_cam_imu"][:3, :3]
    t_ci = cam1["T_cam_imu"][:3, 3]
    X_C = X_I_off @ R_ci.T + t_ci

    uv_pred, J_kb = project_kb_with_jacobian(X_C, cam1)  # (K, 2), (K, 2, 3)
    err_px = uv_pred - uv_meas  # (K, 2)

    # High-angular-rate / motion-blur-aware weighting for visual factors.
    omega_norm = np.linalg.norm(omega_unb, axis=1)
    if omega_rate_scale > 0.0:
        inflation = 1.0 + omega_rate_scale * omega_norm
        sigma_eff = sigma_px * inflation
        huber_eff = huber_delta_px * inflation
    else:
        sigma_eff = sigma_px
        huber_eff = huber_delta_px

    # Huber IRLS weighting + pixel measurement noise scaling
    w_huber = huber_irls_weights(err_px, huber_eff)
    scale = (w_huber / sigma_eff)[:, None]  # (K, 1)

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
    # d(X_I_off)/d(dphi_k) = R_off^T [X_I]_x - dt_eff [v_I]_x
    dX_dphi = np.einsum("kij,kjl->kil", R_off_T, S_XI) - dt_eff[:, None, None] * S_vI
    J_state[:, :, 0:3] = np.einsum("kij,kjl->kil", J_XI_off, dX_dphi)

    # d(X_I_off)/d(dp_k) = -R_off^T R_k^T
    dX_dp = -np.einsum("kij,kjl->kil", R_off_T, R_k_T)
    J_state[:, :, 3:6] = np.einsum("kij,kjl->kil", J_XI_off, dX_dp)

    # d(X_I_off)/d(dv_k) = -dt_eff * R_k^T
    J_state[:, :, 6:9] = np.einsum(
        "kij,kjl->kil", J_XI_off, -dt_eff[:, None, None] * R_k_T
    )

    # d(X_I_off)/d(dbg_k) = -dt_eff * [R_off^T X_I]_x @ C_g
    J_state[:, :, 9:12] = np.einsum(
        "kij,kjl,lm->kim", J_XI_off, -dt_eff[:, None, None] * S_XI_rot, Cg
    )

    # d(uv)/d(X_world) = -d(uv)/d(dp_k)
    J_pt = -J_state[:, :, 3:6]

    # Derivative w.r.t. the global camera-minus-IMU time offset. The
    # rolling-shutter row term is constant, so d(dt_eff)/d(td) = 1.
    # d(R^T X)/dt = R^T [X]_x omega; d(-dt*v)/dt = -v.
    dX_dtd = np.einsum("kij,kjl,kl->ki", R_off_T, S_XI, omega_unb) - v_I
    J_td = np.einsum("kij,kj->ki", J_XI_off, dX_dtd)

    res_weighted = (err_px * scale).ravel()
    J_state *= scale[:, :, None]
    J_pt *= scale[:, :, None]
    J_td *= scale[:, 0][:, None]

    return res_weighted, err_px, J_state, J_pt, J_td


def run_batch_vio_optimization(
    vio_data: dict,
    visual_meas: dict,
    max_iters: int = 40,
    anchor_sigma_px: float = 0.5,
    klt_sigma_px: float = 0.40,
    huber_delta_px: float = 2.5,
    anchor_huber_delta_px: float = 10.0,
    init_prior_rot: float = 0.0,
    init_prior_trans: float = 0.0,
    klt_omega_rate_scale: float = 0.0,
    estimate_time_offset: bool = True,
    time_offset_prior_s: float = 0.005,
) -> dict:
    """
    Runs full-batch on-manifold Levenberg-Marquardt VIO bundle adjustment over all
    frames (IMU poses/velocities/biases), 3D KLT landmarks, 2-DoF gravity tilt,
    and (optionally) the camera-minus-IMU time offset.

    Anchor corners use anchor_sigma_px/anchor_huber_delta_px; KLT uses
    klt_sigma_px/huber_delta_px. When klt_omega_rate_scale > 0, KLT sigma and
    Huber threshold are inflated by (1 + scale * ||omega||) per frame.
    """
    t0 = time.perf_counter()
    N = len(vio_data["t_cam1"])
    cam1 = vio_data["cams"]["cam1"]
    # Rear camera is the only/master camera; observations are already on t_cam1.
    # td is the camera-minus-IMU timing offset, estimated as an extra variable.
    td = 0.0
    TD_MAX = 0.02

    imu_cfg = vio_data.get("imu_cfg", {})
    g_mag = float(imu_cfg.get("g", imu_cfg.get("gravity_magnitude", 9.80665)))
    # imu_preintegration.PreintegratedIMUFactor expects continuous-time noise
    # densities and forms the per-sample covariance as sigma**2 / dt itself.
    # Passing per-sample std here would inflate the variance by another 1000x.
    sigma_g = float(imu_cfg.get("gyro_noise_density", imu_cfg.get("gyroscope_noise_density", 1.8e-4)))
    sigma_a = float(imu_cfg.get("accel_noise_density", imu_cfg.get("accelerometer_noise_density", 2.0e-3)))
    sigma_bg_rw = float(imu_cfg.get("gyro_random_walk", imu_cfg.get("gyroscope_random_walk", 2.0e-5)))
    sigma_ba_rw = float(imu_cfg.get("accel_random_walk", imu_cfg.get("accelerometer_random_walk", 3.0e-4)))
    gyro_correction = np.asarray(
        imu_cfg.get("gyro_correction", np.eye(3, dtype=np.float64)), dtype=np.float64
    ).reshape(3, 3)
    accel_correction = np.asarray(
        imu_cfg.get("accel_correction", np.eye(3, dtype=np.float64)), dtype=np.float64
    ).reshape(3, 3)

    # 1. Preintegrate all 1000 Hz IMU intervals
    b_gyro_0 = np.asarray(vio_data["b_gyro_0"], dtype=np.float64)
    preint_factors = [
        PreintegratedIMUFactor(
            iv, b_gyro_0, np.zeros(3),
            sigma_g=sigma_g, sigma_a=sigma_a,
            gyro_correction=gyro_correction,
            accel_correction=accel_correction,
        )
        for iv in vio_data["imu_intervals_cam1"]
    ]

    # 2. Initialize states, landmarks, and gravity orientation
    R_wi, p_wi, v_w, bg, ba, landmarks, tid_to_lm_idx, R_g = initialize_vio_states(
        vio_data, visual_meas, preint_factors, g_mag=g_mag
    )
    M = len(landmarks)
    omega_raw = extract_frame_gyros(vio_data)
    anchor_packed, klt_packed = pack_visual_observations(visual_meas, tid_to_lm_idx)

    # Optional PnP branch-continuity prior (disabled by default). A very tight
    # anchor weight can keep poses pinned to the initialized PnP branch; a more
    # realistic weight allows the optimizer to smooth noisy tag observations.
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
    td_col = grav_col_offset + 2 if estimate_time_offset else None
    total_vars = grav_col_offset + 2 + (1 if estimate_time_offset else 0)

    def build_system(R_cur, p_cur, v_cur, bg_cur, ba_cur, lm_cur, Rg_cur, td_cur):
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

        # --- Factor A2: Camera/IMU time-offset prior ---
        if estimate_time_offset:
            w_td = 1.0 / max(time_offset_prior_s, 1e-6)
            res_list.append(np.array([td_cur * w_td], dtype=np.float64))
            add_block(row_cursor, td_col, np.array([[w_td]], dtype=np.float64))
            row_cursor += 1

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
        # Weight anchor corners by realistic measurement noise; overly tight sigma
        # can overfit corner jitter and couple rotation into translation.
        res_anc, err_anc_px, J_anc_state, _, J_anc_td = evaluate_visual_residuals_and_jacobians(
            R_cur, p_cur, v_cur, bg_cur, omega_raw,
            anchor_packed["pts_world"],
            anchor_packed["frame_idx"],
            anchor_packed["uv_meas"],
            cam1, td_cur,
            sigma_px=anchor_sigma_px,
            huber_delta_px=anchor_huber_delta_px,
            gyro_correction=gyro_correction,
            omega_rate_scale=0.0,
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
            if estimate_time_offset:
                rows.append(np.broadcast_to(r_base, (2 * K_anc, 1)).ravel())
                cols.append(np.full(2 * K_anc, td_col, dtype=np.int64))
                vals.append(J_anc_td.ravel())
            row_cursor += 2 * K_anc

        # --- Factor D: Rear-Cam KLT Landmarks Bundle Adjustment (cam1) ---
        K_klt = len(klt_packed["frame_idx"])
        err_klt_px = np.zeros((0, 2))
        if K_klt > 0 and M > 0:
            pts_klt_w = lm_cur[klt_packed["lm_idx"]]
            res_klt, err_klt_px, J_klt_state, J_klt_pt, J_klt_td = evaluate_visual_residuals_and_jacobians(
                R_cur, p_cur, v_cur, bg_cur, omega_raw,
                pts_klt_w,
                klt_packed["frame_idx"],
                klt_packed["uv_meas"],
                cam1, td_cur,
                sigma_px=klt_sigma_px,
                huber_delta_px=huber_delta_px,
                gyro_correction=gyro_correction,
                omega_rate_scale=klt_omega_rate_scale,
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

            if estimate_time_offset:
                rows.append(np.broadcast_to(r_base, (2 * K_klt, 1)).ravel())
                cols.append(np.full(2 * K_klt, td_col, dtype=np.int64))
                vals.append(J_klt_td.ravel())
            row_cursor += 2 * K_klt

        r_vec = np.concatenate(res_list)
        J_mat = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row_cursor, total_vars),
        ).tocsc()
        cost = 0.5 * float(np.dot(r_vec, r_vec))
        return cost, r_vec, J_mat, err_anc_px, err_klt_px

    def apply_update(R_cur, p_cur, v_cur, bg_cur, ba_cur, lm_cur, Rg_cur, td_cur, dx: np.ndarray):
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
        if estimate_time_offset:
            td_new = float(np.clip(td_cur + dx[td_col], -TD_MAX, TD_MAX))
        else:
            td_new = 0.0
        return R_new, p_new, v_new, bg_new, ba_new, lm_new, Rg_new, td_new

    # 3. Levenberg-Marquardt Optimization Loop
    cost, r_vec, J_mat, err_anc_0, err_klt_0 = build_system(
        R_wi, p_wi, v_w, bg, ba, landmarks, R_g, td
    )
    err_anc_fin, err_klt_fin = err_anc_0, err_klt_0
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

            # The stiff IMU term makes the normal equations badly scaled across
            # pose, landmark, and bias blocks.  Cap each block per iteration so
            # a single LM step cannot throw landmarks kilometres away before the
            # cost check sees the damage.
            dx = np.asarray(dx, dtype=np.float64).copy()
            for k in range(N):
                for off, limit in ((0, 0.2), (3, 1.0), (6, 5.0), (9, 0.5), (12, 2.0)):
                    block = dx[15 * k + off : 15 * k + off + 3]
                    norm = float(np.linalg.norm(block))
                    if norm > limit:
                        block *= limit / norm
            if M > 0:
                lm_block = dx[n_state_vars:grav_col_offset].reshape(M, 3)
                norms = np.linalg.norm(lm_block, axis=1)
                bad = norms > 5.0
                if np.any(bad):
                    lm_block[bad] *= (5.0 / norms[bad])[:, None]
            dg_block = dx[grav_col_offset : grav_col_offset + 2]
            dg_norm = float(np.linalg.norm(dg_block))
            if dg_norm > 0.5:
                dg_block *= 0.5 / dg_norm
            if estimate_time_offset and abs(float(dx[td_col])) > 0.01:
                dx[td_col] = np.sign(dx[td_col]) * 0.01

            R_try, p_try, v_try, bg_try, ba_try, lm_try, Rg_try, td_try = apply_update(
                R_wi, p_wi, v_w, bg, ba, landmarks, R_g, td, dx
            )
            cost_try, r_try, J_try, err_anc_try, err_klt_try = build_system(
                R_try, p_try, v_try, bg_try, ba_try, lm_try, Rg_try, td_try
            )

            if cost_try < cost:
                step_norm = float(np.linalg.norm(dx))
                R_wi, p_wi, v_w, bg, ba, landmarks, R_g, td = (
                    R_try, p_try, v_try, bg_try, ba_try, lm_try, Rg_try, td_try
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
                R_wi, p_wi, v_w, bg, ba, landmarks, R_g, td
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

    # Assemble 4x4 homogeneous transforms for IMU and rear camera (cam1).
    # The optimizer's world frame is inherited from T_world_tag = I, so gravity
    # is generally not aligned with -Z. Rotate the complete solution into a
    # gravity-aligned world frame, which is also the Z-up frame used by Rerun.
    T_world_imu = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], N, axis=0)
    T_world_imu[:, :3, :3] = R_wi
    T_world_imu[:, :3, 3] = p_wi

    T_imu_cam1 = vio_data["cams"]["cam1"]["T_imu_cam"]
    T_world_cam1 = T_world_imu @ T_imu_cam1[None, :, :]

    R_align = R_g.T
    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R_align
    T_world_imu = T_align[None, :, :] @ T_world_imu
    T_world_cam1 = T_align[None, :, :] @ T_world_cam1
    v_world_out = v_w @ R_align.T
    landmarks_out = landmarks @ R_align.T
    gravity_world = R_align @ (R_g @ np.array([0.0, 0.0, -g_mag], dtype=np.float64))

    return {
        "t_cam1": np.asarray(vio_data["t_cam1"], dtype=np.float64),
        "T_world_imu": T_world_imu,
        "T_world_cam1": T_world_cam1,
        "v_world": v_world_out,
        "b_gyro": bg,
        "b_accel": ba,
        "landmarks_world": landmarks_out,
        "gravity_world": gravity_world,
        "R_gravity_align": R_align,
        "td_cam_imu": float(td),
        "rolling_shutter_readout_s": float(
            cam1.get("rolling_shutter_readout_s") or 0.0
        ),
    }
