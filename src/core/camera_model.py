"""Camera projection/unprojection/Jacobians for KB fisheye and MEI models."""

import numpy as np


def project_kannala_brandt(pts_3d: np.ndarray, cam: dict) -> np.ndarray:
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


def project_kb_with_jacobian(pts_cam: np.ndarray, cam: dict):
    N = len(pts_cam)
    uv = np.empty((N, 2), dtype=np.float64)
    J = np.zeros((N, 2, 3), dtype=np.float64)
    if N == 0:
        return uv, J
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
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


# ---------- MEI (specialist) ----------

def project_mei(pts_cam: np.ndarray, cam: dict) -> np.ndarray:
    uv, _ = project_mei_with_jacobian(pts_cam, cam)
    return uv


def unproject_mei(uv: np.ndarray, cam: dict) -> np.ndarray:
    if len(uv) == 0:
        return np.empty((0, 3), dtype=np.float64)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    k1, k2, k3, p1, p2, xi = cam["k1"], cam["k2"], cam["k3"], cam["p1"], cam["p2"], cam["xi"]
    xd = (uv[:, 0] - cx) / fx
    yd = (uv[:, 1] - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        dx_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        safe_rad = np.where(np.abs(radial) < 1e-12, 1.0, radial)
        xn = (xd - dx_t) / safe_rad
        yn = (yd - dy_t) / safe_rad
        if np.max(np.maximum(np.abs(xn - x), np.abs(yn - y))) < 1e-12:
            x, y = xn, yn
            break
        x, y = xn, yn
    r2 = x * x + y * y
    disc = 1.0 + (1.0 - xi * xi) * r2
    disc = np.where(disc < 0.0, np.nan, disc)
    factor = (xi + np.sqrt(disc)) / (r2 + 1.0)
    bearing = np.stack([factor * x, factor * y, factor - xi], axis=1)
    norms = np.linalg.norm(bearing, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return bearing / norms


def project_mei_with_jacobian(pts_cam: np.ndarray, cam: dict):
    N = len(pts_cam)
    uv = np.full((N, 2), np.nan, dtype=np.float64)
    J = np.zeros((N, 2, 3), dtype=np.float64)
    if N == 0:
        return uv, J
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    k1, k2, k3, p1, p2, xi = cam["k1"], cam["k2"], cam["k3"], cam["p1"], cam["p2"], cam["xi"]

    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    rho = np.sqrt(X * X + Y * Y + Z * Z)
    rho = np.maximum(rho, 1e-12)
    sx, sy, sz = X / rho, Y / rho, Z / rho
    denom = sz + xi
    fold = xi if xi <= 1.0 else 1.0 / xi
    valid = sz > -fold
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    mx = sx / safe_denom
    my = sy / safe_denom
    r2 = mx * mx + my * my
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    drad_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4

    xd = mx * radial + 2.0 * p1 * mx * my + p2 * (r2 + 2.0 * mx * mx)
    yd = my * radial + p1 * (r2 + 2.0 * my * my) + 2.0 * p2 * mx * my

    uv[:, 0] = np.where(valid, fx * xd + cx, np.nan)
    uv[:, 1] = np.where(valid, fy * yd + cy, np.nan)

    # Jacobian
    I3 = np.eye(3)
    for i in range(N):
        if not valid[i]:
            continue
        s = np.array([sx[i], sy[i], sz[i]])
        d = denom[i]
        m = np.array([mx[i], my[i]])
        r2i = r2[i]
        rad = radial[i]
        drad = drad_dr2[i]
        # d(m)/d(s)
        dm_ds = np.array([[1.0 / d, 0.0, -s[0] / (d * d)],
                          [0.0, 1.0 / d, -s[1] / (d * d)]])
        # d(xd,yd)/d(m)
        mx_i, my_i = m
        dxd_dmx = rad + drad * 2 * mx_i * mx_i + 2.0 * p1 * my_i + p2 * (2 * mx_i + 4 * mx_i)
        dxd_dmy = drad * 2 * mx_i * my_i + 2.0 * p1 * mx_i + p2 * (2 * my_i)
        dyd_dmx = drad * 2 * mx_i * my_i + p1 * (2 * mx_i) + 2.0 * p2 * my_i
        dyd_dmy = rad + drad * 2 * my_i * my_i + p1 * (2 * my_i + 4 * my_i) + 2.0 * p2 * mx_i
        dxd_dm = np.array([[dxd_dmx, dxd_dmy], [dyd_dmx, dyd_dmy]])
        # d(s)/d(X)
        rho_i = rho[i]
        ds_dX = (I3 - np.outer(s, s)) / rho_i
        Jm = dxd_dm @ dm_ds @ ds_dX
        J[i, 0, :] = fx * Jm[0]
        J[i, 1, :] = fy * Jm[1]
    return uv, J


def project_with_jacobian(pts_cam: np.ndarray, cam: dict):
    model = cam.get("camera_model", cam.get("model", "kb"))
    if model in ("mei", "unified", "omni"):
        return project_mei_with_jacobian(pts_cam, cam)
    return project_kb_with_jacobian(pts_cam, cam)


def project(pts_cam: np.ndarray, cam: dict) -> np.ndarray:
    model = cam.get("camera_model", cam.get("model", "kb"))
    if model in ("mei", "unified", "omni"):
        return project_mei(pts_cam, cam)
    return project_kannala_brandt(pts_cam, cam)


def unproject(uv: np.ndarray, cam: dict) -> np.ndarray:
    model = cam.get("camera_model", cam.get("model", "kb"))
    if model in ("mei", "unified", "omni"):
        return unproject_mei(uv, cam)
    return unproject_kannala_brandt(uv, cam)
