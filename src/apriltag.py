"""AprilTag geometry, fisheye pose solving and corner refinement..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import cv2
import numpy as np

from .camera_model import (
    project_kannala_brandt,
    unproject_kannala_brandt,
)


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
