"""GTSAM-based full-batch VIO solver.

Replaces the custom LM solver with GTSAM factor graph optimization while
preserving the public ``run_batch_vio_optimization`` interface used by
``optimize.py``.
"""

from __future__ import annotations

import time

import gtsam
import numpy as np
from gtsam import symbol_shorthand as sym

from ..core.camera_model import project_kb_with_jacobian
from ..core.lie_algebra import so3_exp, so3_inv_right_jacobian, so3_log
from .imu_preintegration import PreintegratedIMUFactor
from .vio_initialization import (
    downsample_backend_frames,
    extract_frame_gyros,
    initialize_vio_states,
    interpolate_vio_result,
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
    """Compatibility helper kept for imports; computes residuals/Jacobians like the old solver."""
    K = len(frame_idx)
    if K == 0:
        return (
            np.empty(0),
            np.empty((0, 2)),
            np.empty((0, 2, 15)),
            np.empty((0, 2, 3)),
            np.empty((0, 2)),
        )

    R_k = R_wi[frame_idx]
    p_k = p_wi[frame_idx]
    v_k = v_w[frame_idx]
    bg_k = bg[frame_idx]
    w_k = omega_raw[frame_idx]
    R_k_T = np.transpose(R_k, (0, 2, 1))
    dX_w = pts_world - p_k
    X_I = np.einsum("kij,kj->ki", R_k_T, dX_w)

    omega_unb_raw = w_k - bg_k
    if gyro_correction is None:
        omega_unb = omega_unb_raw
        Cg = np.eye(3, dtype=np.float64)
    else:
        Cg = np.asarray(gyro_correction, dtype=np.float64).reshape(3, 3)
        omega_unb = (Cg @ omega_unb_raw.T).T

    h_px = float(cam1["resolution"][1])
    readout = float(cam1.get("rolling_shutter_readout_s") or 0.0)
    row_fraction = (uv_meas[:, 1] - 0.5 * h_px) / h_px
    dt_eff = float(dt_offset) + row_fraction * max(readout, 0.0)

    R_off = np.array([so3_exp(om * dt) for om, dt in zip(omega_unb, dt_eff)], dtype=np.float64)
    R_off_T = np.transpose(R_off, (0, 2, 1))
    v_I = np.einsum("kij,kj->ki", R_k_T, v_k)
    X_I_off = np.einsum("kij,kj->ki", R_off_T, X_I) - dt_eff[:, None] * v_I

    R_ci = cam1["T_cam_imu"][:3, :3]
    t_ci = cam1["T_cam_imu"][:3, 3]
    X_C = X_I_off @ R_ci.T + t_ci
    uv_pred, J_kb = project_kb_with_jacobian(X_C, cam1)
    err_px = uv_pred - uv_meas

    omega_norm = np.linalg.norm(omega_unb, axis=1)
    if omega_rate_scale > 0.0:
        inflation = 1.0 + omega_rate_scale * omega_norm
        sigma_eff = sigma_px * inflation
        huber_eff = huber_delta_px * inflation
    else:
        sigma_eff = np.full(K, sigma_px, dtype=np.float64)
        huber_eff = np.full(K, huber_delta_px, dtype=np.float64)

    w_huber = huber_irls_weights(err_px, huber_eff)
    scale = (w_huber / sigma_eff)[:, None]
    J_XI_off = np.einsum("kij,jl->kil", J_kb, R_ci)

    def batch_skew(vecs: np.ndarray) -> np.ndarray:
        S = np.zeros((len(vecs), 3, 3), dtype=np.float64)
        S[:, 0, 1], S[:, 0, 2] = -vecs[:, 2], vecs[:, 1]
        S[:, 1, 0], S[:, 1, 2] = vecs[:, 2], -vecs[:, 0]
        S[:, 2, 0], S[:, 2, 1] = -vecs[:, 1], vecs[:, 0]
        return S

    S_XI = batch_skew(X_I)
    S_vI = batch_skew(v_I)
    S_XI_rot = batch_skew(X_I_rot := np.einsum("kij,kj->ki", R_off_T, X_I))

    J_state = np.zeros((K, 2, 15), dtype=np.float64)
    dX_dphi = np.einsum("kij,kjl->kil", R_off_T, S_XI) - dt_eff[:, None, None] * S_vI
    J_state[:, :, 0:3] = np.einsum("kij,kjl->kil", J_XI_off, dX_dphi)
    dX_dp = -np.einsum("kij,kjl->kil", R_off_T, R_k_T)
    J_state[:, :, 3:6] = np.einsum("kij,kjl->kil", J_XI_off, dX_dp)
    J_state[:, :, 6:9] = np.einsum("kij,kjl->kil", J_XI_off, -dt_eff[:, None, None] * R_k_T)
    J_state[:, :, 9:12] = np.einsum("kij,kjl,lm->kim", J_XI_off, -dt_eff[:, None, None] * S_XI_rot, Cg)
    J_pt = -J_state[:, :, 3:6]
    dX_dtd = np.einsum("kij,kjl,kl->ki", R_off_T, S_XI, omega_unb) - v_I
    J_td = np.einsum("kij,kj->ki", J_XI_off, dX_dtd)

    res_weighted = (err_px * scale).ravel()
    J_state *= scale[:, :, None]
    J_pt *= scale[:, :, None]
    J_td *= scale[:, 0][:, None]
    return res_weighted, err_px, J_state, J_pt, J_td


def _make_preintegration_params(
    g_mag: float,
    sigma_g: float,
    sigma_a: float,
    sigma_bg_rw: float = 1e-7,
    sigma_ba_rw: float = 1e-7,
    sigma_int: float = 1e-10,
):
    """Specialist-style combined preintegration params (gravity [0,0,-g])."""
    params = gtsam.PreintegrationCombinedParams.MakeSharedU(g_mag)
    params.setGyroscopeCovariance(np.eye(3) * (sigma_g ** 2))
    params.setAccelerometerCovariance(np.eye(3) * (sigma_a ** 2))
    params.setIntegrationCovariance(np.eye(3) * (sigma_int ** 2))
    params.setBiasAccCovariance(np.eye(3) * (sigma_ba_rw ** 2))
    params.setBiasOmegaCovariance(np.eye(3) * (sigma_bg_rw ** 2))
    return params


def _pose3(R: np.ndarray, p: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(R.copy()), gtsam.Point3(*np.asarray(p, dtype=np.float64).reshape(3)))


def _bias(bg: np.ndarray, ba: np.ndarray) -> gtsam.imuBias.ConstantBias:
    return gtsam.imuBias.ConstantBias(
        np.asarray(ba, dtype=np.float64).reshape(3),
        np.asarray(bg, dtype=np.float64).reshape(3),
    )


def _robust_huber(sigma: float, huber: float):
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(max(float(huber) / max(float(sigma), 1e-8), 1e-6)),
        gtsam.noiseModel.Isotropic.Sigma(2, float(sigma)),
    )


def _project_point_kb(X_c: np.ndarray, cam: dict) -> np.ndarray | None:
    uv, _ = project_kb_with_jacobian(np.asarray(X_c, dtype=np.float64).reshape(1, 3), cam)
    if not np.all(np.isfinite(uv[0])):
        return None
    return uv[0]


def _visual_factor(
    key_x: int,
    key_v: int,
    key_lm: int | None,
    key_td: int | None,
    key_anchor: int | None,
    uv_meas: np.ndarray,
    X_fixed_local: np.ndarray | None,
    cam: dict,
    omega_unb: np.ndarray,
    td_init: float,
    sigma_px: float,
    huber_px: float,
    klt_omega_rate_scale: float = 0.0,
):
    """One rolling-shutter-aware KB reprojection factor.

    ``key_lm`` is set for KLT landmarks. ``key_anchor`` is set for anchor tag
    corners, whose local coordinates are transformed by the optimizable anchor
    pose on every evaluation. Both may not be set at once.
    """
    if (key_lm is None) == (key_anchor is None):
        raise ValueError("A visual factor needs exactly one of key_lm or key_anchor")

    R_ci = np.asarray(cam["T_cam_imu"][:3, :3], dtype=np.float64)
    t_ci = np.asarray(cam["T_cam_imu"][:3, 3], dtype=np.float64)
    h_px = float(cam["resolution"][1])
    readout = float(cam.get("rolling_shutter_readout_s") or 0.0)
    row_fraction = (float(uv_meas[1]) - 0.5 * h_px) / h_px

    om = np.asarray(omega_unb, dtype=np.float64).reshape(3)
    om_norm = float(np.linalg.norm(om))
    if klt_omega_rate_scale > 0.0:
        inflation = 1.0 + klt_omega_rate_scale * om_norm
        sigma_eff = sigma_px * inflation
        huber_eff = huber_px * inflation
    else:
        sigma_eff = sigma_px
        huber_eff = huber_px
    noise = _robust_huber(sigma_eff, huber_eff)

    X_local = (
        np.asarray(X_fixed_local, dtype=np.float64).reshape(3)
        if key_anchor is not None
        else None
    )

    # Fixed key order keeps Jacobian slot lookup simple.
    keys = [key_x, key_v]
    if key_lm is not None:
        keys.append(key_lm)
    if key_td is not None:
        keys.append(key_td)
    if key_anchor is not None:
        keys.append(key_anchor)
    td_pos = keys.index(key_td) if key_td is not None else None
    anchor_pos = keys.index(key_anchor) if key_anchor is not None else None
    lm_pos = keys.index(key_lm) if key_lm is not None else None

    def _skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)

    def _get_td(values: gtsam.Values, this: gtsam.CustomFactor) -> float:
        if td_pos is None:
            return float(td_init)
        return float(values.atVector(this.keys()[td_pos])[0])

    def _world_point(values: gtsam.Values, this: gtsam.CustomFactor) -> np.ndarray:
        if anchor_pos is None:
            return np.asarray(values.atPoint3(this.keys()[lm_pos]), dtype=np.float64)
        T_w_a = np.asarray(values.atPose3(this.keys()[anchor_pos]).matrix(), dtype=np.float64)
        return T_w_a[:3, :3] @ X_local + T_w_a[:3, 3]

    def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list[np.ndarray] | None):
        Xw = _world_point(values, this)
        pose = values.atPose3(this.keys()[0])
        vel = values.atVector(this.keys()[1])
        td_val = _get_td(values, this)
        dt_eff = td_val + row_fraction * max(readout, 0.0)
        R_off_T = so3_exp(om * dt_eff).T

        R_wi = pose.rotation().matrix()
        p_wi = pose.translation()
        X_I = R_wi.T @ (Xw - p_wi)
        v_I = R_wi.T @ vel
        X_I_off = R_off_T @ X_I - dt_eff * v_I
        X_C = R_ci @ X_I_off + t_ci
        uv, J_kb = project_kb_with_jacobian(X_C.reshape(1, 3), cam)
        err = (uv[0] - uv_meas).astype(np.float64)

        if H is not None:
            J_XI_off = J_kb[0] @ R_ci
            S_XI = _skew(X_I)
            S_vI = _skew(v_I)
            dX_dphi = R_off_T @ S_XI - dt_eff * S_vI
            # GTSAM Pose3::retract applies a local-frame translation update:
            # p' = p + R * dt. For X_I = R.T @ (Xw - p), dX_I/dp = -I, but the
            # rolling-shutter rotation offset acts after that: X_I_off involves
            # R_off_T @ X_I, so the pose-translation derivative is -R_off_T.
            dX_dp = -R_off_T
            dX_dv = -dt_eff * R_wi.T
            Hpose = np.zeros((2, 6), dtype=np.float64)
            Hpose[:, 0:3] = J_XI_off @ dX_dphi
            Hpose[:, 3:6] = J_XI_off @ dX_dp
            H[0] = Hpose
            H[1] = J_XI_off @ dX_dv

            if lm_pos is not None:
                # dX_I / dX_w = R_wi^T, then through the offset/extrinsics.
                H[lm_pos] = J_XI_off @ (R_off_T @ R_wi.T)

            if anchor_pos is not None:
                T_w_a = np.asarray(values.atPose3(this.keys()[anchor_pos]).matrix(), dtype=np.float64)
                R_a = T_w_a[:3, :3]
                X_a_rot = R_a @ X_local
                # GTSAM Pose3 retract is a left perturbation:
                # R' = R Exp(w), t' = t + R dt. Therefore
                # d(R p)/dw = -[R p]_x R.
                dXw = np.hstack([-_skew(X_a_rot) @ R_a, R_a])
                dX_I_off = R_off_T @ R_wi.T @ dXw
                H[anchor_pos] = J_XI_off @ dX_I_off

            if td_pos is not None:
                dX_dtd = R_off_T @ (S_XI @ om) - v_I
                H[td_pos] = (J_XI_off @ dX_dtd).reshape(2, 1)

        return err

    return gtsam.CustomFactor(noise, keys, error_func)


def _anchor_gauge_noise(
    world_T_anchor: np.ndarray,
    sigma_gauge: float = 1e-4,
    sigma_tilt: float = 1e3,
) -> gtsam.noiseModel.Base:
    """Tilt-free gauge prior for the reference anchor.

    Position and yaw are pinned tightly to define the world gauge; the two tilt
    axes stay free because gravity makes them observable. Mirrors the
    ``specialist`` anchor gauge prior.
    """
    T = np.asarray(world_T_anchor, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    free = np.eye(3, dtype=np.float64) - np.outer(up, up)
    rot_cov = (sigma_tilt ** 2) * free + (sigma_gauge ** 2) * np.outer(up, up)
    cov = np.zeros((6, 6), dtype=np.float64)
    cov[:3, :3] = R.T @ rot_cov @ R
    cov[3:, 3:] = (sigma_gauge ** 2) * np.eye(3, dtype=np.float64)
    return gtsam.noiseModel.Gaussian.Covariance(cov)


def run_batch_vio_optimization(
    vio_data: dict,
    visual_meas: dict,
    max_iters: int = 30,
    anchor_sigma_px: float = 0.5,
    klt_sigma_px: float = 0.40,
    huber_delta_px: float = 2.5,
    anchor_huber_delta_px: float = 10.0,
    init_prior_rot: float = 0.0,
    init_prior_trans: float = 0.0,
    klt_omega_rate_scale: float = 0.0,
    estimate_time_offset: bool = True,
    time_offset_prior_s: float = 0.005,
    anchor_factor_stride: int = 1,
    max_klt_obs_per_landmark: int | None = None,
    backend_frame_stride: int = 1,
) -> dict:
    """Runs full-batch GTSAM VIO bundle adjustment.

    The external interface matches the previous custom solver. Visual factors are
    GTSAM CustomFactors with KB fisheye projection, rolling-shutter compensation,
    and optional 1-DoF camera-IMU time-offset optimization.
    """
    t0 = time.perf_counter()
    full_vio_data = vio_data
    full_visual_meas = visual_meas
    full_times = np.asarray(vio_data["t_cam1"], dtype=np.float64)
    cam1 = vio_data["cams"]["cam1"]
    td = 0.0  # initial; optimized below if enabled

    imu_cfg = full_vio_data.get("imu_cfg", {})
    g_mag = float(imu_cfg.get("g", imu_cfg.get("gravity_magnitude", 9.80665)))
    sigma_g = float(imu_cfg.get("gyro_noise_density", imu_cfg.get("gyroscope_noise_density", 1.8e-4)))
    sigma_a = float(imu_cfg.get("accel_noise_density", imu_cfg.get("accelerometer_noise_density", 2.0e-3)))
    sigma_bg_rw = float(imu_cfg.get("gyro_random_walk", imu_cfg.get("gyroscope_random_walk", 2.0e-5)))
    sigma_ba_rw = float(imu_cfg.get("accel_random_walk", imu_cfg.get("accelerometer_random_walk", 3.0e-4)))
    gyro_correction = np.asarray(imu_cfg.get("gyro_correction", np.eye(3, dtype=np.float64)), dtype=np.float64).reshape(3, 3)
    accel_correction = np.asarray(imu_cfg.get("accel_correction", np.eye(3, dtype=np.float64)), dtype=np.float64).reshape(3, 3)

    b_gyro_0 = np.asarray(full_vio_data["b_gyro_0"], dtype=np.float64)
    # Intervals from preprocess: gyro has gyro_sign only; accel already has accel_correction.
    # Initialize at full rate first: sparse PnP/IMU alignment is much less stable
    # when the frame interval is artificially enlarged.
    preint_factors = [
        PreintegratedIMUFactor(
            iv, b_gyro_0, np.zeros(3),
            sigma_g=sigma_g, sigma_a=sigma_a,
            gyro_correction=gyro_correction,
            accel_correction=np.eye(3, dtype=np.float64),
        )
        for iv in full_vio_data["imu_intervals_cam1"]
    ]

    R_wi, p_wi, v_w, bg, ba, landmarks, tid_to_lm_idx, R_g = initialize_vio_states(
        full_vio_data, full_visual_meas, preint_factors, g_mag=g_mag
    )

    backend_times = full_times
    stride = max(1, int(backend_frame_stride))
    if stride > 1:
        vio_data, visual_meas, full_times, backend_times, kept = downsample_backend_frames(
            full_vio_data, full_visual_meas, stride
        )
        R_wi = R_wi[kept].copy()
        p_wi = p_wi[kept].copy()
        v_w = v_w[kept].copy()
        bg = bg[kept].copy()
        ba = ba[kept].copy()
        full_lm_dict = full_visual_meas["landmarks_world"]["cam1"]
        reduced_lm_ids = sorted(visual_meas["landmarks_world"]["cam1"])
        landmarks = (
            np.asarray([full_lm_dict[tid] for tid in reduced_lm_ids], dtype=np.float64)
            if reduced_lm_ids
            else np.zeros((0, 3), dtype=np.float64)
        )
        tid_to_lm_idx = {tid: idx for idx, tid in enumerate(reduced_lm_ids)}
        print(
            f"Backend frame stride {stride}: optimizing {len(backend_times)} "
            f"of {len(full_times)} frames; output will be interpolated back to full rate",
            flush=True,
        )

    N = len(vio_data["t_cam1"])
    M = len(landmarks)
    omega_raw = extract_frame_gyros(vio_data)
    anchor_packed, klt_packed = pack_visual_observations(
        visual_meas,
        tid_to_lm_idx,
        anchor_factor_stride=anchor_factor_stride,
        max_klt_obs_per_landmark=max_klt_obs_per_landmark,
    )
    print(
        f"Batch graph: frames={N}, landmarks={M}, "
        f"anchor factors={len(anchor_packed['frame_idx'])}, "
        f"KLT factors={len(klt_packed['frame_idx'])}",
        flush=True,
    )

    # Rotate initial states into gravity-aligned world frame for GTSAM IMU
    # (gravity = [0,0,-g] in navigation frame).
    R_align_init = R_g.T
    g_world_init = R_g @ np.array([0.0, 0.0, -g_mag], dtype=np.float64)
    R_wi0 = np.einsum("ij,kjl->kil", R_align_init, R_wi)
    p_wi0 = (R_align_init @ p_wi.T).T
    v_w0 = (R_align_init @ v_w.T).T
    lm0 = (R_align_init @ landmarks.T).T if M > 0 else landmarks.copy()

    params = _make_preintegration_params(g_mag, sigma_g, sigma_a, sigma_bg_rw=sigma_bg_rw, sigma_ba_rw=sigma_ba_rw)
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    TD_KEY = gtsam.symbol('d', 0)
    td_init = 0.0
    td_key_opt = TD_KEY if estimate_time_offset else None

    # The anchor is an optimizable pose variable. Its initial guess is the
    # frontend's assumed world_T_tag; after rotation into the gravity-aligned
    # GTSAM frame, a tilt-free gauge prior pins its position and yaw only.
    anchor_poses_world = {
        int(tid): np.asarray(T, dtype=np.float64).reshape(4, 4)
        for tid, T in visual_meas.get("anchor_poses_world", {}).items()
    }
    if not anchor_poses_world:
        observed_ids = np.unique(anchor_packed["tag_ids"]) if len(anchor_packed["tag_ids"]) else [0]
        anchor_poses_world = {
            int(tid): np.eye(4, dtype=np.float64) for tid in observed_ids
        }
    if len(anchor_poses_world) != 1:
        raise NotImplementedError(
            "The backend currently supports exactly one anchor pose variable; "
            f"got {sorted(anchor_poses_world)}"
        )
    anchor_key_by_tag = {
        tid: gtsam.symbol('A', tid) for tid in anchor_poses_world
    }

    # Priors: match old solver (bias prior + optional td prior; no X0/V0 prior).
    bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2e-2, 2e-2, 2e-2, 1e-3, 1e-3, 1e-3]))
    graph.add(gtsam.PriorFactorConstantBias(sym.B(0), _bias(gyro_correction @ bg[0], accel_correction @ ba[0]), bias_prior_noise))
    if estimate_time_offset:
        td_sigma = max(float(time_offset_prior_s), 1e-6)
        graph.add(gtsam.PriorFactorVector(TD_KEY, np.array([td_init], dtype=np.float64), gtsam.noiseModel.Isotropic.Sigma(1, td_sigma)))
        values.insert(TD_KEY, np.array([td_init], dtype=np.float64))

    # Anchor variable in the gravity-aligned GTSAM frame.
    for tid, T_w0_a in anchor_poses_world.items():
        T_g_a = np.eye(4, dtype=np.float64)
        T_g_a[:3, :3] = R_align_init @ T_w0_a[:3, :3]
        T_g_a[:3, 3] = R_align_init @ T_w0_a[:3, 3]
        akey = anchor_key_by_tag[tid]
        values.insert(akey, _pose3(T_g_a[:3, :3], T_g_a[:3, 3]))
        graph.add(gtsam.PriorFactorPose3(akey, _pose3(T_g_a[:3, :3], T_g_a[:3, 3]), _anchor_gauge_noise(T_g_a)))

    # Store biases in corrected space in the GTSAM graph: b_corr = C * b_raw.
    for k in range(N):
        values.insert(sym.X(k), _pose3(R_wi0[k], p_wi0[k]))
        values.insert(sym.V(k), v_w0[k].astype(np.float64))
        values.insert(sym.B(k), _bias(gyro_correction @ bg[k], accel_correction @ ba[k]))

    for j in range(M):
        values.insert(sym.L(j), gtsam.Point3(*lm0[j].astype(np.float64)))

    # IMU factors (specialist-style: CombinedImuFactor, zero-order hold).
    for k in range(N - 1):
        # GTSAM subtracts the bias stored in the preintegrated object internally.
        # Work in corrected bias space: b_corr = C * b_raw.
        pim = gtsam.PreintegratedCombinedMeasurements(params, _bias(gyro_correction @ b_gyro_0, accel_correction @ np.zeros(3)))
        iv = vio_data["imu_intervals_cam1"][k]
        dts = iv["dt"]
        gyros = iv["gyro"]
        accels = iv["accel"]
        for i in range(len(dts)):
            dt = float(dts[i])
            if dt <= 0.0:
                continue
            omega_raw_i = gyros[i]
            accel_raw_i = accels[i]
            # Gyro: preprocess applied gyro_sign only; apply gyro_correction here.
            # Accel: preprocess already applied accel_correction; do NOT reapply.
            omega = gyro_correction @ omega_raw_i
            accel = np.asarray(accel_raw_i, dtype=np.float64)
            pim.integrateMeasurement(accel, omega.astype(np.float64), dt)
        graph.add(gtsam.CombinedImuFactor(sym.X(k), sym.V(k), sym.X(k + 1), sym.V(k + 1), sym.B(k), sym.B(k + 1), pim))

    # Optional PnP continuity priors
    init_pnp = visual_meas.get("init_poses_world_imu", {})
    if init_prior_rot > 0.0 or init_prior_trans > 0.0:
        rot_sig = init_prior_rot if init_prior_rot > 0.0 else 1e6
        trans_sig = init_prior_trans if init_prior_trans > 0.0 else 1e6
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([rot_sig, rot_sig, rot_sig, trans_sig, trans_sig, trans_sig]))
        for k, T_wi_init in init_pnp.items():
            ki = int(k)
            if 0 <= ki < N:
                T_aligned = np.eye(4)
                T_aligned[:3, :3] = R_align_init @ T_wi_init[:3, :3]
                T_aligned[:3, 3] = R_align_init @ T_wi_init[:3, 3]
                graph.add(gtsam.PriorFactorPose3(sym.X(ki), _pose3(T_aligned[:3, :3], T_aligned[:3, 3]), prior_noise))

    # Anchor factors. Corners stay in the tag frame; the solver estimates the
    # tag pose instead of treating it as a fixed world point.
    anc = anchor_packed
    for i in range(len(anc["frame_idx"])):
        k = int(anc["frame_idx"][i])
        tid = int(anc["tag_ids"][i])
        uv = anc["uv_meas"][i].astype(np.float64)
        X_local = anc["pts_local"][i].astype(np.float64)
        om_unb = gyro_correction @ (omega_raw[k] - bg[k]) if gyro_correction is not None else (omega_raw[k] - bg[k])
        graph.add(_visual_factor(
            sym.X(k), sym.V(k), None, td_key_opt, anchor_key_by_tag[tid],
            uv, X_local, cam1, om_unb, td_init,
            anchor_sigma_px, anchor_huber_delta_px, 0.0,
        ))

    # KLT factors
    klt = klt_packed
    for i in range(len(klt["frame_idx"])):
        k = int(klt["frame_idx"][i])
        j = int(klt["lm_idx"][i])
        uv = klt["uv_meas"][i].astype(np.float64)
        om_unb = gyro_correction @ (omega_raw[k] - bg[k]) if gyro_correction is not None else (omega_raw[k] - bg[k])
        graph.add(_visual_factor(
            sym.X(k), sym.V(k), sym.L(j), td_key_opt, None,
            uv, None, cam1, om_unb, td_init,
            klt_sigma_px, huber_delta_px, klt_omega_rate_scale,
        ))

    # Optimize
    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(int(max_iters))
    lm_params.setRelativeErrorTol(1e-5)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    result = optimizer.optimize()
    elapsed = time.perf_counter() - t0
    if estimate_time_offset:
        td = float(result.atVector(TD_KEY)[0])
    print(f"GTSAM batch VIO optimization finished in {elapsed:.2f} s. final error={graph.error(result):.3f} td={td:.6f}s")

    # Extract results
    R_out = np.zeros((N, 3, 3), dtype=np.float64)
    p_out = np.zeros((N, 3), dtype=np.float64)
    v_out = np.zeros((N, 3), dtype=np.float64)
    bg_out = np.zeros((N, 3), dtype=np.float64)
    ba_out = np.zeros((N, 3), dtype=np.float64)
    Cg_inv = np.linalg.inv(gyro_correction)
    Ca_inv = np.linalg.inv(accel_correction)
    for k in range(N):
        pose = result.atPose3(sym.X(k))
        R_out[k] = pose.rotation().matrix()
        p_out[k] = pose.translation()
        v_out[k] = result.atVector(sym.V(k))
        b = result.atConstantBias(sym.B(k))
        bg_out[k] = Cg_inv @ b.gyroscope()
        ba_out[k] = Ca_inv @ b.accelerometer()

    lm_out = np.zeros((M, 3), dtype=np.float64)
    for j in range(M):
        lm_out[j] = result.atPoint3(sym.L(j))

    # Rotate back to original world frame, then gravity-align output like old solver
    R_align_inv = R_align_init.T
    R_wi_world = np.einsum("ij,kjl->kil", R_align_inv, R_out)
    p_wi_world = (R_align_inv @ p_out.T).T
    v_world_out0 = (R_align_inv @ v_out.T).T
    lm_world0 = (R_align_inv @ lm_out.T).T if M > 0 else lm_out

    # Final gravity alignment: use initial R_g to align to Z-up gravity frame
    R_align = R_g.T
    T_world_imu = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], N, axis=0)
    T_world_imu[:, :3, :3] = R_wi_world
    T_world_imu[:, :3, 3] = p_wi_world
    T_imu_cam1 = vio_data["cams"]["cam1"]["T_imu_cam"]
    T_world_cam1 = T_world_imu @ T_imu_cam1[None, :, :]
    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R_align
    T_world_imu = T_align[None, :, :] @ T_world_imu
    T_world_cam1 = T_align[None, :, :] @ T_world_cam1

    # Optimized anchor poses are already expressed in the final gravity-aligned
    # world frame used by T_world_imu / T_world_cam1 above.
    anchor_poses_out = {
        tid: np.asarray(result.atPose3(akey).matrix(), dtype=np.float64).copy()
        for tid, akey in anchor_key_by_tag.items()
    }

    result = {
        "t_cam1": np.asarray(backend_times, dtype=np.float64),
        "T_world_imu": T_world_imu,
        "T_world_cam1": T_world_cam1,
        "v_world": v_world_out0 @ R_align.T,
        "b_gyro": bg_out,
        "b_accel": ba_out,
        "landmarks_world": lm_world0 @ R_align.T if M > 0 else lm_world0,
        "gravity_world": R_align @ (R_g @ np.array([0.0, 0.0, -g_mag], dtype=np.float64)),
        "R_gravity_align": R_align,
        "td_cam_imu": float(td),
        "rolling_shutter_readout_s": float(cam1.get("rolling_shutter_readout_s") or 0.0),
        # Single-anchor run: keep this a plain numeric array so np.savez/np.load
        # with allow_pickle=False keeps working.
        "T_world_tag": next(iter(anchor_poses_out.values())),
    }
    if stride > 1:
        result = interpolate_vio_result(result, backend_times, full_times)
    return result
