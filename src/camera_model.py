"""Kannala-Brandt fisheye projection, unprojection and Jacobians..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np


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
