"""Gyro-aided 2-point epipolar outlier rejection..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np


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
