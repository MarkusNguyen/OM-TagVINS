import warnings

import yaml
import cv2
import numpy as np
import rerun as rr


# The gyro stream in the .rrd uses the opposite sign convention from the
# right-hand body-frame angular velocity expected by so3_exp() in
# src/gyro.py and src/imu_preintegration.py. Negate it once here so bias
# estimation, gyro-aided tracking and IMU preintegration all agree.
GYRO_SIGN = -1.0


def load_calibration(yaml_path: str, actual_wh: tuple[int, int]) -> tuple[dict, dict]:
    """
    Loads calib.yaml and automatically scales focal length & principal point
    if the recorded frame resolution (actual_wh = (width, height)) differs
    from the calibration resolution (e.g., 2880x2880 vs 1920x1920).
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    actual_w, actual_h = actual_wh
    cams = {}

    for name in ("cam0", "cam1"):
        c = cfg[name]
        cal_w, cal_h = c["resolution"]
        scale_x = actual_w / float(cal_w)
        scale_y = actual_h / float(cal_h)

        if not np.isclose(scale_x, scale_y, rtol=1e-3):
            raise ValueError(
                f"{name}: non-uniform scaling ({scale_x:.4f} vs {scale_y:.4f}) "
                "breaks Kannala-Brandt radial distortion assumptions."
            )

        fx, fy, cx, cy = c["intrinsics"]
        T_cam_imu = np.array(c["T_cam_imu"], dtype=np.float64)

        cams[name] = {
            "resolution": (actual_w, actual_h),
            "scale": scale_x,
            "fx": fx * scale_x,
            "fy": fy * scale_y,
            "cx": cx * scale_x,
            "cy": cy * scale_y,
            "D": np.array(c["distortion_coeffs"], dtype=np.float64),
            "timeshift": float(c["timeshift_cam_imu"]),
            "T_cam_imu": T_cam_imu,
            "T_imu_cam": np.linalg.inv(T_cam_imu),  # Body-from-camera for GTSAM
        }

    imu_cfg = dict(cfg["imu0"])
    # Optional fixed gyro scale/misalignment correction:
    #   omega_corrected = C_g @ (omega_raw - b_g)
    if "gyro_correction" in imu_cfg:
        Cg = np.array(imu_cfg["gyro_correction"], dtype=np.float64).reshape(3, 3)
    else:
        scale = np.array(imu_cfg.get("gyro_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        Cg = np.diag(scale)
        if "gyro_misalignment" in imu_cfg:
            Cg = Cg @ np.array(imu_cfg["gyro_misalignment"], dtype=np.float64).reshape(3, 3)
    imu_cfg["gyro_correction"] = Cg

    return cams, imu_cfg


def _unwrap_image_cell(cell):
    """Unwraps Rerun's nested object/list containers down to the raw buffer."""
    obj = cell
    for _ in range(8):
        if isinstance(obj, np.ndarray):
            if obj.dtype != object:
                return obj
            if obj.size == 0:
                break
            obj = obj.reshape(-1)[0]
        elif isinstance(obj, memoryview):
            return np.frombuffer(obj, dtype=np.uint8)
        elif isinstance(obj, (bytes, bytearray)):
            return np.frombuffer(obj, dtype=np.uint8)
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                break
            obj = obj[0]
        else:
            break
    return np.asarray(obj)


def _decode_image_cell(cell) -> np.ndarray:
    """Decodes a JPEG-encoded image cell or converts a raw Rerun image to BGR."""
    arr = np.ascontiguousarray(_unwrap_image_cell(cell))

    if arr.ndim == 1 and arr.dtype == np.uint8:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode JPEG image blob from Rerun.")
        return img

    if arr.ndim == 3 and arr.dtype == np.uint8:
        # Rerun image buffers are RGB(A); OpenCV expects BGR.
        if arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    if arr.ndim == 2 and arr.dtype == np.uint8:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unexpected image cell shape={arr.shape}, dtype={arr.dtype}")


class LazyImageSequence:
    """Decodes Rerun image blobs on demand so all frames are never held in RAM."""

    def __init__(self, cells, cache_size: int = 2):
        self._cells = list(cells)
        self._cache_size = max(1, int(cache_size))
        self._cache: dict[int, np.ndarray] = {}
        self._order: list[int] = []

    def __len__(self) -> int:
        return len(self._cells)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        if index < 0:
            index += len(self._cells)
        if index < 0 or index >= len(self._cells):
            raise IndexError(index)

        if index in self._cache:
            return self._cache[index]

        img = _decode_image_cell(self._cells[index])
        self._cache[index] = img
        self._order.append(index)
        if len(self._order) > self._cache_size:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        return img


def _timeline_to_seconds(column) -> np.ndarray:
    """Converts a Rerun pandas timeline (datetime/timedelta/float) to seconds."""
    values = column.to_numpy()
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64) * 1e-9
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[ns]").astype(np.int64) * 1e-9

    seconds = values.astype(np.float64)
    finite = seconds[np.isfinite(seconds)]
    if finite.size and float(np.max(np.abs(finite))) > 1e6:
        # Numeric Rerun timelines are commonly stored in nanoseconds.
        seconds = seconds * 1e-9
    return seconds


def load_base_rrd(rrd_path: str, timeline: str = "time") -> dict:
    """
    Reads synchronized camera frames and raw IMU channels from base.rrd.
    Adjust `timeline` if your recording uses a custom timeline name (e.g., 'timestamp').
    """
    with rr.server.Server(datasets={"rec": [rrd_path]}) as server:
        dataset = server.client().get_dataset("rec")

        # 1. Query IMU streams
        imu_paths = [
            "/imu/accel/x", "/imu/accel/y", "/imu/accel/z",
            "/imu/gyro/x",  "/imu/gyro/y",  "/imu/gyro/z",
        ]
        imu_df = (
            dataset.filter_contents(imu_paths)
            .reader(index=timeline, fill_latest_at=True)
            .to_pandas()
        )

        # dropna() on the whole frame also removes log_time/frame (always null for
        # columns), deleting every IMU sample. Only require the scalar columns.
        imu_cols = []
        for entity in imu_paths:
            matches = [c for c in imu_df.columns if c.startswith(entity + ":")]
            if not matches:
                raise ValueError(f"Missing IMU component {entity} in {rrd_path}")
            imu_cols.append(matches[0])
        imu_df = imu_df.dropna(subset=imu_cols).sort_values(timeline)
        if imu_df.empty:
            raise ValueError(f"No valid IMU samples found in {rrd_path}")

        t_imu = _timeline_to_seconds(imu_df[timeline])

        def _col(entity: str) -> np.ndarray:
            match = [c for c in imu_df.columns if c.startswith(entity + ":")][0]
            vals = imu_df[match].apply(lambda v: v[0] if isinstance(v, (list, np.ndarray)) else v)
            return vals.to_numpy(dtype=np.float64)

        accel_raw = np.column_stack([_col("/imu/accel/x"), _col("/imu/accel/y"), _col("/imu/accel/z")])
        gyro_raw  = GYRO_SIGN * np.column_stack([_col("/imu/gyro/x"),  _col("/imu/gyro/y"),  _col("/imu/gyro/z")])

        # 2. Query the rear camera stream only.
        cam_df = (
            dataset.filter_contents(["/camera/rear"])
            .reader(index=timeline)
            .to_pandas()
            .sort_values(timeline)
        )

        rear_col = [c for c in cam_df.columns if c.startswith("/camera/rear:")][0]
        cam_df = cam_df.dropna(subset=[rear_col])
        if cam_df.empty:
            raise ValueError(f"No rear-camera frames found in {rrd_path}")

        t_cam_raw = _timeline_to_seconds(cam_df[timeline])

        # Keep encoded blobs (cheap) and decode frames lazily on access.
        rear_imgs = LazyImageSequence(cam_df[rear_col].tolist())

    return {
        "t_cam_raw": t_cam_raw,
        "rear_imgs": rear_imgs,
        "t_imu": t_imu,
        "accel_raw": accel_raw,
        "gyro_raw": gyro_raw,
    }


def initialize_static_imu(
    t_imu: np.ndarray,
    accel_ms2: np.ndarray,
    gyro_rad_s: np.ndarray,
    t_first_frame: float,
    std_thresh: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses the stationary pre-roll before t_first_frame (filtering out button-tap
    ripples) to estimate initial gyro bias and gravity-aligned orientation R_I_G.
    """
    mask_pre = t_imu < t_first_frame
    t_pre, a_pre, w_pre = t_imu[mask_pre], accel_ms2[mask_pre], gyro_rad_s[mask_pre]

    if len(t_pre) < 10:
        raise ValueError(
            f"Only {len(t_pre)} IMU samples before the first frame "
            f"(t={t_first_frame:.6f}s); cannot estimate initial biases."
        )

    diffs = np.diff(t_pre)
    dt = float(np.median(diffs)) if len(diffs) else 0.001
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.001

    win = min(len(w_pre), max(10, int(round(0.25 / dt))))
    step = max(1, win // 2)

    windows = []
    for i in range(0, len(w_pre) - win + 1, step):
        w_chunk = w_pre[i : i + win]
        a_chunk = a_pre[i : i + win]
        windows.append((float(np.max(np.std(w_chunk, axis=0))), w_chunk, a_chunk))
    if not windows:
        windows = [(float(np.max(np.std(w_pre, axis=0))), w_pre, a_pre)]

    accepted = [item for item in windows if item[0] < std_thresh]
    if not accepted:
        # Do not silently return zero bias/identity: use the quietest pre-roll
        # window, which is still far better than the default values.
        accepted = [min(windows, key=lambda item: item[0])]
        warnings.warn(
            f"No IMU window passed std_thresh={std_thresh}; falling back to the "
            f"quietest window (max gyro std={accepted[0][0]:.5f} rad/s).",
            RuntimeWarning,
            stacklevel=2,
        )

    b_gyro_0 = np.mean(np.vstack([item[1] for item in accepted]), axis=0)
    a_mean = np.mean(np.vstack([item[2] for item in accepted]), axis=0)

    # Align measured upward specific force (+Z_world) with IMU axes
    a_norm = float(np.linalg.norm(a_mean))
    if not np.isfinite(a_norm) or a_norm < 1e-6:
        raise ValueError("Invalid mean accelerometer reading during static initialization.")
    z_w = a_mean / a_norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_w = np.cross(z_w, ref)
    y_w /= np.linalg.norm(y_w)
    x_w = np.cross(y_w, z_w)
    R_I_G_0 = np.column_stack([x_w, y_w, z_w])

    return b_gyro_0, R_I_G_0


def slice_imu_intervals(
    t_nodes: np.ndarray,
    t_imu: np.ndarray,
    accel_ms2: np.ndarray,
    gyro_rad_s: np.ndarray,
) -> list[dict]:
    """
    Trims unpaired pre-roll/post-roll IMU outside [t_nodes[0], t_nodes[-1]] and
    slices into N-1 intervals [t_nodes[k], t_nodes[k+1]] with exact interpolated endpoints.
    """
    if len(t_nodes) < 2:
        raise ValueError("At least two camera timestamps are required.")
    if t_imu[0] > t_nodes[0] + 1e-9 or t_imu[-1] < t_nodes[-1] - 1e-9:
        raise ValueError("IMU stream does not fully cover the shifted camera timestamps.")

    a_nodes = np.column_stack([np.interp(t_nodes, t_imu, accel_ms2[:, j]) for j in range(3)])
    g_nodes = np.column_stack([np.interp(t_nodes, t_imu, gyro_rad_s[:, j]) for j in range(3)])

    intervals = []
    for k in range(len(t_nodes) - 1):
        t0, t1 = t_nodes[k], t_nodes[k + 1]
        inside = (t_imu > t0) & (t_imu < t1)

        t_seg = np.concatenate([[t0], t_imu[inside], [t1]])
        a_seg = np.vstack([a_nodes[k], accel_ms2[inside], a_nodes[k + 1]])
        g_seg = np.vstack([g_nodes[k], gyro_rad_s[inside], g_nodes[k + 1]])

        intervals.append({
            "t0": t0,
            "t1": t1,
            "timestamps": t_seg,
            "dt": np.diff(t_seg),
            "accel": a_seg,
            "gyro": g_seg,
        })
    return intervals


def prepare_vio_inputs(rrd_path: str, yaml_path: str, timeline: str = "time") -> dict:
    raw = load_base_rrd(rrd_path, timeline=timeline)

    # Check 1: Auto-detect rear-frame resolution and scale intrinsics.
    first_rear = raw["rear_imgs"][0]
    if first_rear is None or first_rear.size == 0:
        raise ValueError("Could not decode the first rear-camera frame.")
    h, w = first_rear.shape[:2]
    cams, imu_cfg = load_calibration(yaml_path, actual_wh=(w, h))

    # Check 2: Convert accelerometer g -> m/s^2 (with guard against double conversion)
    g_mag = float(imu_cfg["gravity_magnitude"])
    med_norm = np.median(np.linalg.norm(raw["accel_raw"][:500], axis=1))
    accel_ms2 = raw["accel_raw"] * g_mag if med_norm < 3.0 else raw["accel_raw"].copy()

    # Check 3: Apply per-camera hardware timeshift_cam_imu
    t_cam1 = raw["t_cam_raw"] + cams["cam1"]["timeshift"]  # Rear (_00_)

    # Static initialization uses the rear-camera timeline too. The VIO itself
    # is rear-only, so do not introduce a front-camera (cam0) time shift here.
    t_first_frame = raw["t_cam_raw"][0] + cams["cam1"]["timeshift"]
    b_gyro_0, R_I_G_0 = initialize_static_imu(
        raw["t_imu"], accel_ms2, raw["gyro_raw"], t_first_frame=t_first_frame
    )

    # Check 4: Trim unpaired IMU and slice exact [t_k, t_{k+1}] intervals for the rear camera.
    imu_intervals_cam1 = slice_imu_intervals(t_cam1, raw["t_imu"], accel_ms2, raw["gyro_raw"])

    return {
        "cams": cams,
        "imu_cfg": imu_cfg,
        "t_cam1": t_cam1,
        "rear_imgs": raw["rear_imgs"],
        "b_gyro_0": b_gyro_0,
        "R_I_G_0": R_I_G_0,
        "imu_intervals_cam1": imu_intervals_cam1,
    }