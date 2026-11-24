"""VIO state, landmark and gravity initialization..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from ..core.lie_algebra import so3_exp, so3_log
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

    if not init_pnp:
        raise RuntimeError(
            "No AprilTag anchor PnP poses are available; cannot initialize the "
            "VIO world frame. Check anchor visibility and tag configuration."
        )

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

    # Central-difference initial velocities, then a linear visual-inertial
    # refinement that also uses preintegrated IMU deltas.
    for k in range(N):
        if k == 0:
            v_w[k] = (p_wi[1] - p_wi[0]) / max(t_cam1[1] - t_cam1[0], 1e-4)
        elif k == N - 1:
            v_w[k] = (p_wi[N - 1] - p_wi[N - 2]) / max(t_cam1[N - 1] - t_cam1[N - 2], 1e-4)
        else:
            v_w[k] = (p_wi[k + 1] - p_wi[k - 1]) / max(t_cam1[k + 1] - t_cam1[k - 1], 1e-4)
    v_w = refine_initial_velocities(t_cam1, R_wi, p_wi, preint_factors, g_world, v_w)

    # Pack static 3D landmarks into contiguous array
    lm_dict = visual_meas["landmarks_world"]["cam1"]
    lm_ids = sorted(lm_dict.keys())
    tid_to_lm_idx = {tid: idx for idx, tid in enumerate(lm_ids)}
    landmarks = np.array([lm_dict[tid] for tid in lm_ids], dtype=np.float64) if lm_ids else np.zeros((0, 3))

    return R_wi, p_wi, v_w, bg, ba, landmarks, tid_to_lm_idx, R_g


def refine_initial_velocities(
    t_cam1: np.ndarray,
    R_wi: np.ndarray,
    p_wi: np.ndarray,
    preint_factors: list[PreintegratedIMUFactor],
    g_world: np.ndarray,
    v_init: np.ndarray,
) -> np.ndarray:
    """Linear VI alignment of initial frame velocities.

    Uses preintegrated position/velocity deltas together with the PnP pose graph:
        p_j - p_i - 0.5*g*dt^2 - R_i*dp = v_i*dt
        v_j - v_i = g*dt + R_i*dv
    A weak finite-difference prior keeps the least-squares system well posed.
    """
    N = len(t_cam1)
    if N < 2 or len(preint_factors) < N - 1:
        return v_init.copy()

    I3 = np.eye(3, dtype=np.float64)
    n_unknown = 3 * N
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rhs: list[float] = []
    next_row = 0

    def add_block_row(blocks, rhs_vec):
        """Appends one 3-row block equation; each entry is (frame_index, 3x3 block)."""
        nonlocal next_row
        for frame_idx, block in blocks:
            for a in range(3):
                for b in range(3):
                    value = float(block[a, b])
                    if value == 0.0:
                        continue
                    row_indices.append(next_row + a)
                    col_indices.append(3 * frame_idx + b)
                    values.append(value)
        rhs.extend(float(v) for v in rhs_vec)
        next_row += 3

    for k in range(N - 1):
        pf = preint_factors[k]
        dt = max(float(pf.dt), 1e-4)
        Rk = R_wi[k]

        b_pos = (
            p_wi[k + 1] - p_wi[k] - 0.5 * g_world * (dt * dt) - Rk @ pf.dp
        ) / dt
        add_block_row([(k, I3)], b_pos)
        add_block_row([(k, -I3), (k + 1, I3)], g_world * dt + Rk @ pf.dv)

    # Weak prior towards central differences (guards against degenerate motion).
    for k in range(N):
        add_block_row([(k, 0.1 * I3)], 0.1 * v_init[k])

    # The dense equivalent is (3N x 3N) and needs several GB for long recordings.
    # Every row has only 3 or 6 nonzeros, so keep it sparse and solve with LSQR.
    A = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(next_row, n_unknown),
        dtype=np.float64,
    ).tocsr()
    b = np.asarray(rhs, dtype=np.float64)
    v_sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    return v_sol.reshape(N, 3)


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


def _merge_imu_intervals(intervals: list[dict], start: int, end: int) -> dict:
    """Merges consecutive IMU intervals into one interval spanning two kept frames."""
    if end <= start + 1:
        return intervals[start]

    timestamps, accel, gyro = [], [], []
    for idx in range(start, end):
        segment = intervals[idx]
        skip = 0 if idx == start else 1  # adjacent intervals share one endpoint
        timestamps.extend(segment["timestamps"][skip:])
        accel.extend(segment["accel"][skip:])
        gyro.extend(segment["gyro"][skip:])
    times = np.asarray(timestamps, dtype=np.float64)
    return {
        "t0": float(times[0]),
        "t1": float(times[-1]),
        "timestamps": times,
        "dt": np.diff(times),
        "accel": np.asarray(accel, dtype=np.float64),
        "gyro": np.asarray(gyro, dtype=np.float64),
    }


def downsample_backend_frames(vio_data: dict, visual_meas: dict, stride: int):
    """Builds a smaller backend problem by keeping every N-th camera frame.

    Front-end outputs stay at full rate; only the batch optimizer uses the
    reduced state. Frame indices in the measurements are remapped accordingly.
    """
    stride = max(1, int(stride))
    t_cam1 = np.asarray(vio_data["t_cam1"], dtype=np.float64)
    n_frames = len(t_cam1)
    if stride == 1 or n_frames < 3:
        return (
            vio_data,
            visual_meas,
            t_cam1,
            t_cam1,
            np.arange(n_frames, dtype=np.int64),
        )

    kept = np.arange(0, n_frames, stride, dtype=np.int64)
    if kept[-1] != n_frames - 1:
        kept = np.append(kept, n_frames - 1)
    old_to_new = {int(old): int(new) for new, old in enumerate(kept)}

    new_vio = dict(vio_data)
    new_vio["t_cam1"] = t_cam1[kept]
    new_vio["imu_intervals_cam1"] = [
        _merge_imu_intervals(vio_data["imu_intervals_cam1"], int(kept[i]), int(kept[i + 1]))
        for i in range(len(kept) - 1)
    ]

    new_anchor_obs = []
    for obs in visual_meas["anchor_obs"]:
        new_idx = old_to_new.get(int(obs["frame_idx"]))
        if new_idx is None:
            continue
        remapped = dict(obs)
        remapped["frame_idx"] = new_idx
        new_anchor_obs.append(remapped)

    new_init_poses = {}
    for frame_idx, pose in visual_meas["init_poses_world_imu"].items():
        new_idx = old_to_new.get(int(frame_idx))
        if new_idx is not None:
            new_init_poses[new_idx] = pose

    new_tracks = {}
    for cam_name, tracks in visual_meas["klt_tracks"].items():
        new_tracks[cam_name] = {}
        for tid, obs_list in tracks.items():
            remapped_obs = []
            for obs in obs_list:
                new_idx = old_to_new.get(int(obs[0]))
                if new_idx is None:
                    continue
                if len(obs) == 6:
                    remapped_obs.append((new_idx, obs[1], obs[2], obs[3], obs[4], obs[5]))
                else:
                    remapped_obs.append((new_idx, obs[1], obs[2]))
            if remapped_obs:
                new_tracks[cam_name][tid] = remapped_obs

    # Drop landmarks that no longer have any observation in the reduced graph;
    # otherwise GTSAM would insert unconstrained Point3 variables.
    lm_by_cam = {
        cam: dict(landmarks)
        for cam, landmarks in visual_meas.get("landmarks_world", {}).items()
    }
    referenced = set(new_tracks.get("cam1", {}))
    if "cam1" in lm_by_cam:
        lm_by_cam["cam1"] = {
            tid: xyz for tid, xyz in lm_by_cam["cam1"].items() if tid in referenced
        }

    new_visual = dict(visual_meas)
    new_visual["anchor_obs"] = new_anchor_obs
    new_visual["init_poses_world_imu"] = new_init_poses
    new_visual["klt_tracks"] = new_tracks
    new_visual["landmarks_world"] = lm_by_cam
    return new_vio, new_visual, t_cam1, t_cam1[kept], kept


def pack_visual_observations(
    visual_meas: dict,
    tid_to_lm_idx: dict,
    anchor_factor_stride: int = 1,
    max_klt_obs_per_landmark: int | None = None,
):
    """Flattens anchor tag corner observations and KLT track observations into vectorized arrays."""
    # Anchor tag corners are stored in the tag's local frame. The backend adds
    # an optimizable anchor pose variable and transforms them into the world
    # frame inside each visual factor.
    anc_frames, anc_tag_ids, anc_pts_local, anc_pts_w, anc_uv = [], [], [], [], []
    anchor_obs = visual_meas["anchor_obs"]
    factor_stride = max(1, int(anchor_factor_stride))
    if factor_stride > 1:
        subsampled = [
            obs for obs in anchor_obs if int(obs["frame_idx"]) % factor_stride == 0
        ]
        if subsampled:
            anchor_obs = subsampled
    for obs in anchor_obs:
        f_idx = int(obs["frame_idx"])
        tag_id = int(obs["tag_id"])
        local = np.asarray(obs.get("corners_local"), dtype=np.float64).reshape(-1, 3)
        world = np.asarray(obs["corners_world"], dtype=np.float64).reshape(-1, 3)
        for pt_l, pt_w, uv in zip(local, world, obs["corners_uv"]):
            anc_frames.append(f_idx)
            anc_tag_ids.append(tag_id)
            anc_pts_local.append(pt_l)
            anc_pts_w.append(pt_w)
            anc_uv.append(uv)

    anchor_packed = {
        "frame_idx": np.asarray(anc_frames, dtype=np.int64),
        "tag_ids": np.asarray(anc_tag_ids, dtype=np.int64),
        "pts_local": np.asarray(anc_pts_local, dtype=np.float64).reshape(-1, 3),
        "pts_world": np.asarray(anc_pts_w, dtype=np.float64).reshape(-1, 3),
        "uv_meas": np.asarray(anc_uv, dtype=np.float64).reshape(-1, 2),
    }

    # KLT tracks (optimizable 3D landmarks)
    klt_frames, klt_lm_idx, klt_uv = [], [], []
    klt_tracks = visual_meas["klt_tracks"]["cam1"]
    for tid, lm_idx in tid_to_lm_idx.items():
        obs_list = klt_tracks[tid]
        if max_klt_obs_per_landmark and len(obs_list) > int(max_klt_obs_per_landmark):
            sample_idx = np.linspace(
                0, len(obs_list) - 1, int(max_klt_obs_per_landmark), dtype=np.int64
            )
            obs_list = [obs_list[int(i)] for i in sample_idx]
        for obs in obs_list:
            f_idx = int(obs[0])
            if len(obs) == 6:
                # Compact (frame, u, v, rx, ry, rz) observation.
                uv = (obs[1], obs[2])
            else:
                uv = obs[1]
            klt_frames.append(f_idx)
            klt_lm_idx.append(lm_idx)
            klt_uv.append(uv)

    klt_packed = {
        "frame_idx": np.asarray(klt_frames, dtype=np.int64),
        "lm_idx": np.asarray(klt_lm_idx, dtype=np.int64),
        "uv_meas": np.asarray(klt_uv, dtype=np.float64).reshape(-1, 2),
    }

    return anchor_packed, klt_packed


def _interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolates a rigid transform with geodesic rotation and linear translation."""
    R0, p0 = T0[:3, :3], T0[:3, 3]
    R1, p1 = T1[:3, :3], T1[:3, 3]
    dR = R0 @ so3_exp(float(alpha) * so3_log(R0.T @ R1))
    dp = (1.0 - float(alpha)) * p0 + float(alpha) * p1
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = dR
    T[:3, 3] = dp
    return T


def interpolate_vio_result(result: dict, t_backend: np.ndarray, t_full: np.ndarray) -> dict:
    """Expands a downsampled backend result back to the full camera frame rate."""
    t_backend = np.asarray(t_backend, dtype=np.float64)
    t_full = np.asarray(t_full, dtype=np.float64)
    if len(t_backend) == len(t_full) and np.array_equal(t_backend, t_full):
        return result

    if len(t_backend) < 2:
        raise ValueError("At least two backend timestamps are required for interpolation")

    right = np.searchsorted(t_backend, t_full, side="right") - 1
    i0 = np.clip(right, 0, len(t_backend) - 2)
    alpha = np.clip(
        (t_full - t_backend[i0]) / np.maximum(t_backend[i0 + 1] - t_backend[i0], 1e-12),
        0.0,
        1.0,
    )

    def interp_pose_array(key: str) -> np.ndarray:
        src = np.asarray(result[key], dtype=np.float64)
        return np.stack(
            [
                _interpolate_pose(src[i0[i]], src[i0[i] + 1], alpha[i])
                for i in range(len(t_full))
            ],
            axis=0,
        )

    def interp_vector_array(key: str) -> np.ndarray:
        src = np.asarray(result[key], dtype=np.float64)
        return (1.0 - alpha[:, None]) * src[i0] + alpha[:, None] * src[i0 + 1]

    expanded = dict(result)
    expanded["t_cam1"] = t_full.copy()
    expanded["T_world_imu"] = interp_pose_array("T_world_imu")
    expanded["T_world_cam1"] = interp_pose_array("T_world_cam1")
    for key in ("v_world", "b_gyro", "b_accel"):
        if key in result:
            expanded[key] = interp_vector_array(key)
    return expanded
