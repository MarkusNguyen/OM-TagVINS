"""SO(3) Lie-group helpers and skew-symmetric matrices..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Returns 3x3 skew-symmetric matrix [v]_x."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) -> so(3) rotation vector."""
    cos_theta = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
    if np.pi - theta < 1e-5:
        # Near-pi fallback using eigenvector of eigenvalue +1
        eigvals, eigvecs = np.linalg.eig(R)
        axis = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
        axis /= np.linalg.norm(axis)
        return theta * axis
    factor = 0.5 * theta / np.sin(theta)
    return factor * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian J_r(phi) of SO(3)."""
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-7:
        return np.eye(3, dtype=np.float64) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    return (
        np.eye(3, dtype=np.float64)
        - ((1.0 - np.cos(theta)) / t2) * K
        + ((theta - np.sin(theta)) / (t2 * theta)) * (K @ K)
    )


def so3_inv_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian J_r^{-1}(phi) of SO(3)."""
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-7:
        return np.eye(3, dtype=np.float64) + 0.5 * K + (1.0 / 12.0) * (K @ K)
    t2 = theta * theta
    cot_half = np.cos(0.5 * theta) / np.sin(0.5 * theta)
    coeff = (1.0 / t2) - (cot_half / (2.0 * theta))
    return np.eye(3, dtype=np.float64) + 0.5 * K + coeff * (K @ K)


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
