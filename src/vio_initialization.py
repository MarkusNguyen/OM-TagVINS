"""VIO state, landmark and gravity initialization..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np

from .imu_preintegration import PreintegratedIMUFactor


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

    # Estimate initial gravity direction in World frame. Prefer the static
    # pre-roll estimate (IMU -> gravity-aligned frame) composed with the first
    # PnP pose; fall back to early recording intervals only if unavailable.
    first_k = min(init_pnp.keys())
    R0 = init_pnp[first_k][:3, :3]
    R_I_G_0 = vio_data.get("R_I_G_0")

    if R_I_G_0 is not None:
        R_g = R0 @ np.asarray(R_I_G_0, dtype=np.float64)
    else:
        accel_samples = []
        for k in range(min(15, len(imu_intervals))):
            accel_samples.append(imu_intervals[k]["accel"])
        mean_accel_I = np.mean(np.vstack(accel_samples), axis=0)
        # Specific force at rest: a_I = -R_IW * g_W  =>  g_W_dir = -R_WI * a_I / ||a_I||
        g_dir_w = -(R0 @ mean_accel_I)
        g_dir_w /= np.linalg.norm(g_dir_w)

        # Construct orthonormal R_g such that R_g @ [0, 0, -1] == g_dir_w
        z_col = -g_dir_w
        ref = (
            np.array([1.0, 0.0, 0.0])
            if abs(z_col[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
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
