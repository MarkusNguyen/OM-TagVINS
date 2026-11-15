import argparse
import cv2
import numpy as np
from pupil_apriltags import Detector

from preprocess import prepare_vio_inputs
from settings import DEFAULT_CONFIG_PATH, load_run_config


# The visual front-end processes the rear camera only.
REAR_CAM = "cam1"
ACTIVE_CAMS = (REAR_CAM,)
CAMERA_IMAGE_KEYS = {REAR_CAM: "rear_imgs"}


# ==============================================================================
# 1. Lie Group SO(3) & IMU Gyroscope Rotation Integration
# ==============================================================================

def so3_exp(omega_dt: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3) via Rodrigues' formula."""
    angle = float(np.linalg.norm(omega_dt))
    if angle < 1e-10:
        wx, wy, wz = omega_dt
        return np.array(
            [[1.0, -wz, wy], [wz, 1.0, -wx], [-wy, wx, 1.0]],
            dtype=np.float64,
        )
    axis = omega_dt / angle
    x, y, z = axis
    K = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def integrate_gyro_rotation(interval: dict, b_gyro: np.ndarray) -> np.ndarray:
    """
    Integrates bias-corrected 1000 Hz gyroscope measurements across [t_{k-1}, t_k]
    using midpoint/trapezoidal integration to return R_{I_{k-1}, I_k}.
    """
    dts = interval["dt"]
    gyros = interval["gyro"]
    R_prev_curr = np.eye(3, dtype=np.float64)

    for j in range(len(dts)):
        omega_mid = 0.5 * (gyros[j] + gyros[j + 1]) - b_gyro
        R_prev_curr = R_prev_curr @ so3_exp(omega_mid * float(dts[j]))

    return R_prev_curr


def compute_camera_relative_rotation(R_I_prev_curr: np.ndarray, cam: dict) -> np.ndarray:
    """
    Converts relative IMU rotation R_{I_{k-1}, I_k} into camera frame rotation
    dR_{C_k, C_{k-1}} mapping bearing rays from frame k-1 into frame k:
      f_{C_k} = R_CI * R_{I_{k-1}, I_k}^T * R_CI^T * f_{C_{k-1}}
    """
    R_ci = cam["T_cam_imu"][:3, :3]
    return R_ci @ R_I_prev_curr.T @ R_ci.T


# ==============================================================================
# 2. Kannala-Brandt (Equidistant) Fisheye Camera Model
# ==============================================================================

def project_kannala_brandt(pts_3d: np.ndarray, cam: dict) -> np.ndarray:
    """
    Projects 3D points/rays in the camera frame (N, 3) -> 2D fisheye pixels (N, 2)
    using r_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8).
    """
    if len(pts_3d) == 0:
        return np.empty((0, 2), dtype=np.float64)

    x, y, z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    r_xy = np.hypot(x, y)
    theta = np.arctan2(r_xy, z)

    k1, k2, k3, k4 = cam["D"]
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    rd = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)

    scale = np.ones_like(r_xy)
    np.divide(rd, r_xy, out=scale, where=r_xy > 1e-10)
    u = cam["fx"] * (x * scale) + cam["cx"]
    v = cam["fy"] * (y * scale) + cam["cy"]
    return np.column_stack([u, v])


def unproject_kannala_brandt(uv: np.ndarray, cam: dict) -> np.ndarray:
    """
    Inverts the Kannala-Brandt radial polynomial via Newton-Raphson iterations
    to map 2D fisheye pixels (N, 2) -> 3D unit bearing rays (N, 3) on S^2.
    """
    if len(uv) == 0:
        return np.empty((0, 3), dtype=np.float64)

    mx = (uv[:, 0] - cam["cx"]) / cam["fx"]
    my = (uv[:, 1] - cam["cy"]) / cam["fy"]
    rd = np.hypot(mx, my)
    k1, k2, k3, k4 = cam["D"]

    theta = np.minimum(rd, np.pi * 0.5)
    for _ in range(12):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        f = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - rd
        df = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
        theta -= f / np.maximum(df, 1e-8)

    scale = np.ones_like(rd)
    np.divide(np.sin(theta), rd, out=scale, where=rd > 1e-10)
    rays = np.stack([mx * scale, my * scale, np.cos(theta)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


# ==============================================================================
# 3. AprilTag Geometry & Fisheye Pose Solver
# ==============================================================================

def get_tag_corners_3d(tag_size_m: float) -> np.ndarray:
    """
    Returns the 4 corner coordinates in the local tag frame (Z=0 plane) matching
    pupil_apriltags' counter-clockwise corner order.
    """
    h = 0.5 * tag_size_m
    return np.array(
        [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
        dtype=np.float64,
    )


def solve_tag_pose_fisheye(
    corners_uv: np.ndarray,
    cam: dict,
    tag_size_m: float,
    prev_T_cam_tag: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Undistorts fisheye tag corners onto the unit pinhole plane and solves T_cam_tag.

    A planar square tag can have two IPPE solutions with similar reprojection
    error. When a previous pose is available, select the solution that is closest
    to it; otherwise fall back to the lowest reprojection error.
    """
    corners_uv = np.asarray(corners_uv, dtype=np.float64).reshape(-1, 2)
    if corners_uv.shape[0] != 4 or not np.all(np.isfinite(corners_uv)):
        return None

    rays = unproject_kannala_brandt(corners_uv, cam)
    if not np.all(np.isfinite(rays)):
        return None
    if np.any(rays[:, 2] <= 0.05):
        return None

    norm_xy = (rays[:, :2] / rays[:, 2:3]).astype(np.float64)
    obj_pts = get_tag_corners_3d(tag_size_m)

    n_solutions, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
        obj_pts,
        norm_xy,
        cameraMatrix=np.eye(3, dtype=np.float64),
        distCoeffs=None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if n_solutions < 1:
        return None

    candidates = []
    for rvec, tvec, reproj_err in zip(rvecs, tvecs, reproj_errors):
        tvec = np.asarray(tvec, dtype=np.float64).ravel()
        if tvec.shape != (3,) or not np.all(np.isfinite(tvec)) or tvec[2] <= 0.0:
            continue
        R, _ = cv2.Rodrigues(rvec)
        if not np.all(np.isfinite(R)):
            continue
        T_cam_tag = np.eye(4, dtype=np.float64)
        T_cam_tag[:3, :3] = R
        T_cam_tag[:3, 3] = tvec
        candidates.append((T_cam_tag, float(np.asarray(reproj_err).ravel()[0])))

    if not candidates:
        return None
    if len(candidates) == 1 or prev_T_cam_tag is None:
        return min(candidates, key=lambda item: item[1])[0]

    prev_R = prev_T_cam_tag[:3, :3]
    prev_t = prev_T_cam_tag[:3, 3]
    depth = max(float(np.linalg.norm(prev_t)), 0.1)

    def continuity_score(item):
        T_cam_tag, _ = item
        cos_angle = 0.5 * (np.trace(prev_R.T @ T_cam_tag[:3, :3]) - 1.0)
        rot_err = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        trans_err = float(np.linalg.norm(T_cam_tag[:3, 3] - prev_t))
        return rot_err + trans_err / depth

    return min(candidates, key=continuity_score)[0]


def refine_tag_corners_virtual_pinhole(
    gray: np.ndarray,
    rough_corners_uv: np.ndarray,
    tag_id: int,
    cam: dict,
    detector,
    patch_size: int = 240,
) -> np.ndarray:

    rays = unproject_kannala_brandt(rough_corners_uv, cam)
    z_c = np.mean(rays, axis=0)
    z_c /= np.linalg.norm(z_c)

    ref = np.array([0.0, 1.0, 0.0]) if abs(z_c[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_c = np.cross(ref, z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R_cam_virt = np.column_stack([x_c, y_c, z_c])

    rays_v = rays @ R_cam_virt
    u_norm = rays_v[:, 0] / np.maximum(rays_v[:, 2], 1e-6)
    v_norm = rays_v[:, 1] / np.maximum(rays_v[:, 2], 1e-6)
    max_ext = max(float(np.max(np.abs(u_norm))), float(np.max(np.abs(v_norm)))) * 1.45
    if max_ext <= 1e-4:
        return rough_corners_uv

    half = 0.5 * patch_size
    f_virt = half / max_ext

    grid = np.arange(patch_size, dtype=np.float64)
    uu, vv = np.meshgrid(grid, grid)
    X_v = np.stack([(uu - half) / f_virt, (vv - half) / f_virt, np.ones_like(uu)], axis=-1)
    X_c = X_v.reshape(-1, 3) @ R_cam_virt.T

    uv_fish = project_kannala_brandt(X_c, cam).reshape(patch_size, patch_size, 2)
    patch = cv2.remap(
        gray,
        uv_fish[:, :, 0].astype(np.float32),
        uv_fish[:, :, 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    dets = detector.detect(patch)
    for d in dets:
        if d.tag_id == tag_id:
            c_p = np.asarray(d.corners, dtype=np.float64)
            pts_v = np.column_stack([
                (c_p[:, 0] - half) / f_virt,
                (c_p[:, 1] - half) / f_virt,
                np.ones(len(c_p), dtype=np.float64),
            ])
            pts_c = pts_v @ R_cam_virt.T
            return project_kannala_brandt(pts_c, cam)

    return rough_corners_uv


# ==============================================================================
# 4. Static Foreground & Aperture Masking
# ==============================================================================

def build_static_masks(cam: dict, cam_name: str) -> np.ndarray:
    """
    Builds a uint8 mask (255 = valid static background, 0 = ignored):
      - Blocks outer black fisheye border beyond active sensor circle
      - Blocks lower gripper/hand region in cam0 (front)
      - Blocks operator body/arms in cam1 (rear)
    """
    w, h = cam["resolution"]
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - cam["cx"], yy - cam["cy"])
    mask = ((dist <= 0.465 * w) * 255).astype(np.uint8)

    if cam_name == "cam0":
        gripper_poly = np.array(
            [[[0.10 * w, 0.52 * h], [0.90 * w, 0.52 * h], [0.96 * w, h], [0.04 * w, h]]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, gripper_poly, 0)
    elif cam_name == "cam1":
        operator_poly = np.array(
            [[[0.50 * w, 0.20 * h], [w, 0.20 * h], [w, h], [0.32 * w, h]]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, operator_poly, 0)

    return mask


# ==============================================================================
# 5. Gyro-Aided 2-Point Epipolar RANSAC for Dynamic Outlier Rejection
# ==============================================================================

def ransac_2pt_gyro_epipolar(
    r0: np.ndarray,
    r1: np.ndarray,
    dR_c1_c0: np.ndarray,
    ang_thresh_rad: float = 0.003,
    n_iters: int = 120,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    Rejects dynamic outliers on unit bearing rays using known gyro rotation dR_c1_c0:
      - Under pure rotation / stationary camera: checks ||angle(r1, dR * r0)|| < thresh
      - Under 6-DoF motion: 2-point RANSAC on t_hat . (r1 x (dR * r0)) = 0
      - Rejects points rigidly carried with the camera/gripper when the scene translates
    """
    N = len(r0)
    if N < 4:
        return np.ones(N, dtype=bool)

    r0_rot = (dR_c1_c0 @ r0.T).T
    rot_err = np.arccos(np.clip(np.sum(r1 * r0_rot, axis=1), -1.0, 1.0))
    med_parallax = float(np.median(rot_err))

    # Case 1: Pure rotation or stationary camera
    if med_parallax < ang_thresh_rad * 1.5:
        return rot_err < (ang_thresh_rad * 2.0)

    # Case 2: Camera is translating -> 2-point RANSAC for unit translation direction t_hat
    c = np.cross(r1, r0_rot)
    c_norms = np.linalg.norm(c, axis=1)

    # When camera translates strongly, points with near-zero parallax are carried rigidly
    # with the camera/gripper rather than belonging to the static world
    rigid_with_cam = (med_parallax > 0.010) & (rot_err < 0.18 * med_parallax)
    valid_sample = (~rigid_with_cam) & (c_norms > 1e-5)
    if np.sum(valid_sample) < 4:
        return rot_err < (ang_thresh_rad * 2.5)

    c_unit = np.zeros_like(c)
    c_unit[valid_sample] = c[valid_sample] / c_norms[valid_sample, None]

    rng = np.random.default_rng(rng_seed)
    valid_idx = np.flatnonzero(valid_sample)
    best_inliers = np.zeros(N, dtype=bool)
    best_count = -1

    for _ in range(n_iters):
        i, j = rng.choice(valid_idx, size=2, replace=False)
        t_cand = np.cross(c_unit[i], c_unit[j])
        t_norm = np.linalg.norm(t_cand)
        if t_norm < 1e-8:
            continue
        t_hat = t_cand / t_norm

        epipolar_err = np.abs(c_unit @ t_hat)
        inliers = valid_sample & (epipolar_err < ang_thresh_rad)
        cnt = int(np.sum(inliers))
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    return best_inliers


# ==============================================================================
# 6. IMU-Aided Fisheye KLT Optical Flow Tracker
# ==============================================================================

class IMUAidedFisheyeKLTTracker:
    """
    Tracks static background features on fisheye frames using:
      1) Gyro-integrated rotation prediction (OPTFLOW_USE_INITIAL_FLOW)
      2) Forward-backward Lucas-Kanade sub-pixel refinement
      3) 2-point Gyro-aided epipolar RANSAC for dynamic outlier rejection
    """

    def __init__(self, cam: dict, static_mask: np.ndarray, max_corners: int = 250):
        self.cam = cam
        self.static_mask = static_mask
        self.max_corners = max_corners
        self.next_track_id = 0
        self.prev_gray = None
        self.prev_pts = np.empty((0, 2), dtype=np.float32)
        self.prev_rays = np.empty((0, 3), dtype=np.float64)
        self.prev_ids = np.empty((0,), dtype=np.int64)
        self.tracks = {}  # track_id -> list of (frame_idx, uv, unit_ray)

    def _predict_initial_pixels(self, dR_c1_c0: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
        """Rotates previous 3D bearing rays by dR_c1_c0 and projects onto current fisheye frame."""
        h, w = img_shape
        rays_pred = (dR_c1_c0 @ self.prev_rays.T).T
        uv_pred = project_kannala_brandt(rays_pred, self.cam)

        out_of_bounds = (
            ~np.isfinite(uv_pred[:, 0])
            | ~np.isfinite(uv_pred[:, 1])
            | (uv_pred[:, 0] < 0) | (uv_pred[:, 0] >= w)
            | (uv_pred[:, 1] < 0) | (uv_pred[:, 1] >= h)
        )
        uv_pred[out_of_bounds] = self.prev_pts[out_of_bounds]
        return uv_pred.astype(np.float32)

    def _detect_new_features(self, gray: np.ndarray, existing_pts: np.ndarray) -> np.ndarray:
        need = self.max_corners - len(existing_pts)
        if need <= 10:
            return np.empty((0, 2), dtype=np.float32)

        det_mask = self.static_mask.copy()
        for u, v in existing_pts:
            cv2.circle(det_mask, (int(round(u)), int(round(v))), 18, 0, -1)

        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=need, qualityLevel=0.01, minDistance=18, mask=det_mask
        )
        if corners is None:
            return np.empty((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def step(
        self,
        frame_idx: int,
        gray: np.ndarray,
        dR_c1_c0: np.ndarray | None = None,
    ):
        h, w = gray.shape

        if self.prev_gray is not None and len(self.prev_pts) > 0:
            p0 = self.prev_pts.reshape(-1, 1, 2)

            if dR_c1_c0 is not None:
                p1_init = self._predict_initial_pixels(dR_c1_c0, (h, w)).reshape(-1, 1, 2)
                p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    gray,
                    p0,
                    p1_init,
                    winSize=(21, 21),
                    maxLevel=3,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                )
            else:
                p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3
                )

            p0_back_init = p0.copy()
            p0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self.prev_gray,
                p1,
                p0_back_init,
                winSize=(21, 21),
                maxLevel=3,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )

            p1_2d = p1.reshape(-1, 2)
            fb_err = np.linalg.norm(self.prev_pts - p0_back.reshape(-1, 2), axis=1)
            in_bounds = (
                (st_fwd.ravel() == 1)
                & (st_bwd.ravel() == 1)
                & (fb_err < 1.0)
                & (p1_2d[:, 0] >= 0) & (p1_2d[:, 0] < w)
                & (p1_2d[:, 1] >= 0) & (p1_2d[:, 1] < h)
            )
            in_mask = np.array(
                [self.static_mask[int(v), int(u)] > 0 if ok else False for (u, v), ok in zip(p1_2d, in_bounds)],
                dtype=bool,
            )
            valid = in_bounds & in_mask

            r0_v = self.prev_rays[valid]
            pts1_v = p1_2d[valid]
            ids_v = self.prev_ids[valid]
            r1_v = unproject_kannala_brandt(pts1_v, self.cam)

            # Geometric outlier & dynamic object rejection via 2-point Gyro RANSAC
            if dR_c1_c0 is not None and len(r0_v) >= 6:
                keep = ransac_2pt_gyro_epipolar(r0_v, r1_v, dR_c1_c0)
                pts1_v = pts1_v[keep]
                r1_v = r1_v[keep]
                ids_v = ids_v[keep]
        else:
            pts1_v = np.empty((0, 2), dtype=np.float32)
            r1_v = np.empty((0, 3), dtype=np.float64)
            ids_v = np.empty((0,), dtype=np.int64)

        # Replenish static background corners up to max_corners
        new_pts = self._detect_new_features(gray, pts1_v)
        new_rays = unproject_kannala_brandt(new_pts, self.cam)
        new_ids = np.arange(self.next_track_id, self.next_track_id + len(new_pts), dtype=np.int64)
        self.next_track_id += len(new_pts)

        cur_pts = np.vstack([pts1_v, new_pts]) if len(new_pts) else pts1_v
        cur_rays = np.vstack([r1_v, new_rays]) if len(new_rays) else r1_v
        cur_ids = np.concatenate([ids_v, new_ids]) if len(new_ids) else ids_v

        for tid, uv, ray in zip(cur_ids, cur_pts, cur_rays):
            self.tracks.setdefault(int(tid), []).append((frame_idx, uv.copy(), ray.copy()))

        self.prev_gray = gray
        self.prev_pts = cur_pts
        self.prev_rays = cur_rays
        self.prev_ids = cur_ids


# ==============================================================================
# 7. Wide-Angle Bearing-Ray Triangulation for Initial 3D Landmarks
# ==============================================================================

def triangulate_bearing_rays(T_cam_worlds: list[np.ndarray], rays_cam: list[np.ndarray]) -> np.ndarray | None:
    """
    Linear DLT triangulation directly on 3D unit bearing rays:
      [ray_cam]_x * P_cam_world * [X_world; 1] = 0
    Avoids pinhole z=0 division so features near or beyond 90 deg FOV triangulate cleanly.
    """
    rows = []
    for T_cw, f in zip(T_cam_worlds, rays_cam):
        if not np.all(np.isfinite(T_cw)) or not np.all(np.isfinite(f)):
            return None
        P = T_cw[:3, :]
        fx, fy, fz = f
        skew = np.array(
            [[0.0, -fz, fy], [fz, 0.0, -fx], [-fy, fx, 0.0]],
            dtype=np.float64,
        )
        rows.append(skew @ P)

    A = np.vstack(rows)
    if not np.all(np.isfinite(A)):
        return None
    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1]
    if not np.all(np.isfinite(X_hom)) or abs(X_hom[3]) < 1e-10:
        return None
    return X_hom[:3] / X_hom[3]


def triangulate_klt_tracks(
    klt_tracks: dict,
    init_poses_world_imu: dict[int, np.ndarray],
    cams: dict,
    min_parallax_deg: float = 1.0,
    max_reproj_px: float = 3.0,
) -> dict:
    """Triangulates initial 3D world coordinates for rear-camera KLT tracks."""
    landmarks_world = {cam_name: {} for cam_name in ACTIVE_CAMS}
    min_cos = np.cos(np.deg2rad(min_parallax_deg))

    for cam_name in ACTIVE_CAMS:
        cam = cams[cam_name]
        T_cam_imu = cam["T_cam_imu"]

        for tid, obs_list in klt_tracks[cam_name].items():
            posed_obs = [
                (f_idx, uv, ray) for (f_idx, uv, ray) in obs_list if f_idx in init_poses_world_imu
            ]
            if len(posed_obs) < 3:
                continue

            T_cws, rays_c, rays_w, uvs = [], [], [], []
            for f_idx, uv, ray in posed_obs:
                T_wi = init_poses_world_imu[f_idx]
                T_cw = T_cam_imu @ np.linalg.inv(T_wi)
                R_wc = T_wi[:3, :3] @ cam["T_imu_cam"][:3, :3]
                T_cws.append(T_cw)
                rays_c.append(ray)
                rays_w.append(R_wc @ ray)
                uvs.append(uv)

            # Use the maximum pairwise parallax rather than first/last only:
            # a track can start and end at nearly the same bearing while
            # passing through a much wider baseline in between.
            rays_w_arr = np.asarray(rays_w, dtype=np.float64)
            cos_matrix = np.clip(rays_w_arr @ rays_w_arr.T, -1.0, 1.0)
            if float(np.min(cos_matrix)) > min_cos:
                continue

            X_w = triangulate_bearing_rays(T_cws, rays_c)
            if X_w is None:
                continue

            valid = True
            for T_cw, uv_meas in zip(T_cws, uvs):
                X_c = T_cw[:3, :3] @ X_w + T_cw[:3, 3]
                # The linear DLT is sign-ambiguous for nearly parallel rays and
                # can return the antipodal intersection behind the cameras.
                if not np.all(np.isfinite(X_c)) or X_c[2] <= 0.0:
                    valid = False
                    break
                uv_proj = project_kannala_brandt(X_c[None, :], cam)[0]
                if not np.all(np.isfinite(uv_proj)):
                    valid = False
                    break
                if np.linalg.norm(uv_proj - uv_meas) > max_reproj_px:
                    valid = False
                    break

            if valid:
                landmarks_world[cam_name][tid] = X_w

    return landmarks_world


# ==============================================================================
# 8. Main Front-End Pipeline Entrypoint
# ==============================================================================

def extract_visual_measurements(
    vio_data: dict,
    anchor_tags: dict,
    max_corners_per_cam: int = 220,
    min_track_len: int = 5,
) -> dict:
    """
    Runs stationary anchor AprilTag detection and IMU-aided fisheye KLT tracking
    on the rear camera across all synchronized frames.
    """
    cams = vio_data["cams"]
    cam1 = cams[REAR_CAM]
    b_gyro_0 = vio_data["b_gyro_0"]
    imu_intervals_cam1 = vio_data["imu_intervals_cam1"]

    detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=1)

    trackers = {
        cam_name: IMUAidedFisheyeKLTTracker(
            cams[cam_name],
            build_static_masks(cams[cam_name], cam_name),
            max_corners_per_cam,
        )
        for cam_name in ACTIVE_CAMS
    }

    anchor_corners_world = {}
    for tid, info in anchor_tags.items():
        local_4x3 = get_tag_corners_3d(info["size_m"])
        local_hom = np.hstack([local_4x3, np.ones((4, 1), dtype=np.float64)])
        anchor_corners_world[tid] = (info["T_world_tag"] @ local_hom.T).T[:, :3]

    anchor_obs = []
    init_poses_world_imu = {}
    last_tag_poses = {}

    num_frames = len(vio_data["rear_imgs"])
    for k in range(num_frames):
        for cam_name in ACTIVE_CAMS:
            bgr = vio_data[CAMERA_IMAGE_KEYS[cam_name]][k]
            cam = cams[cam_name]

            # Integrate 1000 Hz gyro between frame k-1 and k on the rear
            # camera's timeshifted timeline.
            R_I_prev_curr = (
                integrate_gyro_rotation(imu_intervals_cam1[k - 1], b_gyro_0)
                if k > 0
                else None
            )

            gray_rear = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            dets = detector.detect(gray_rear)

            for det in dets:
                tid = int(det.tag_id)
                if tid not in anchor_tags:
                    continue

                corners_uv = refine_tag_corners_virtual_pinhole(
                    gray_rear, np.asarray(det.corners, dtype=np.float64), det.tag_id, cam1, detector
                )
                info = anchor_tags[tid]
                rays = unproject_kannala_brandt(corners_uv, cam)
                anchor_obs.append({
                    "frame_idx": k,
                    "cam_name": cam_name,
                    "tag_id": tid,
                    "corners_uv": corners_uv,
                    "corners_rays": rays,
                    "corners_world": anchor_corners_world[tid],
                })

                T_cam_tag = solve_tag_pose_fisheye(
                    corners_uv, cam, info["size_m"], prev_T_cam_tag=last_tag_poses.get(tid)
                )
                if T_cam_tag is not None:
                    last_tag_poses[tid] = T_cam_tag
                    if k not in init_poses_world_imu:
                        T_world_imu = (
                            info["T_world_tag"]
                            @ np.linalg.inv(T_cam_tag)
                            @ cam["T_cam_imu"]
                        )
                        init_poses_world_imu[k] = T_world_imu

            dR_c1_c0 = compute_camera_relative_rotation(R_I_prev_curr, cam) if R_I_prev_curr is not None else None
            trackers[cam_name].step(k, gray_rear, dR_c1_c0=dR_c1_c0)

    klt_tracks = {
        cam_name: {tid: obs for tid, obs in trk.tracks.items() if len(obs) >= min_track_len}
        for cam_name, trk in trackers.items()
    }

    landmarks_world = triangulate_klt_tracks(klt_tracks, init_poses_world_imu, cams)

    return {
        "anchor_obs": anchor_obs,
        "init_poses_world_imu": init_poses_world_imu,
        "klt_tracks": klt_tracks,
        "landmarks_world": landmarks_world,
    }


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_run_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Run the rear-camera IMU-aided VIO visual front-end on a base .rrd recording"
    )
    parser.add_argument(
        "--config",
        default=config_args.config,
        help=f"VIO YAML configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--rrd",
        required=True,
        help="Path to the raw base .rrd recording",
    )
    parser.add_argument(
        "--calib",
        default="config/cam-imu.yaml",
        help="Path to the camera/IMU calibration YAML",
    )
    parser.add_argument(
        "--anchor-id",
        type=int,
        default=cfg["anchor"]["id"],
        help="Stationary anchor tag36h11 ID",
    )
    parser.add_argument(
        "--anchor-size",
        type=float,
        default=cfg["anchor"]["size_m"],
        help="Anchor tag side length in metres",
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
    args = parser.parse_args()

    print(f"VIO config: {args.config}")

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

    print(f"Frames processed:        {len(vio_data['t_cam1'])}")
    print(f"Anchor observations:     {len(visual_meas['anchor_obs'])}")
    print(f"Initial PnP poses:       {len(visual_meas['init_poses_world_imu'])}")
    print(
        f"Triangulated KLT (rear): "
        f"{len(visual_meas['landmarks_world'][REAR_CAM])}"
    )