"""Bearing-ray triangulation for initial 3D landmarks..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np

from ..core.camera_model import project_kannala_brandt
from ..core.constants import ACTIVE_CAMS


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
    min_parallax_deg: float = 0.5,
    max_reproj_px: float = 3.0,
    max_obs_per_track: int | None = None,
) -> dict:
    """Triangulates initial 3D world coordinates for rear-camera KLT tracks."""
    landmarks_world = {cam_name: {} for cam_name in ACTIVE_CAMS}
    min_cos = np.cos(np.deg2rad(min_parallax_deg))

    for cam_name in ACTIVE_CAMS:
        cam = cams[cam_name]
        T_cam_imu = cam["T_cam_imu"]

        for tid, obs_list in klt_tracks[cam_name].items():
            posed_obs = []
            for obs in obs_list:
                f_idx = int(obs[0])
                if f_idx not in init_poses_world_imu:
                    continue
                if len(obs) == 6:
                    # Compact observation from IMUAidedFisheyeKLTTracker.
                    uv = (obs[1], obs[2])
                    ray = (obs[3], obs[4], obs[5])
                else:
                    uv, ray = obs[1], obs[2]
                posed_obs.append(
                    (
                        f_idx,
                        np.asarray(uv, dtype=np.float64),
                        np.asarray(ray, dtype=np.float64),
                    )
                )
            if len(posed_obs) < 3:
                continue

            # A long track can contain thousands of nearly redundant rays. Their
            # DLT system adds little information but costs a lot of SVD time.
            if max_obs_per_track and len(posed_obs) > int(max_obs_per_track):
                sample_idx = np.linspace(
                    0, len(posed_obs) - 1, int(max_obs_per_track), dtype=np.int64
                )
                posed_obs = [posed_obs[int(i)] for i in sample_idx]
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
