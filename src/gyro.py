"""Gyroscope rotation integration and camera-relative rotations..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import numpy as np

from .lie_algebra import so3_exp


def integrate_gyro_rotation(
    interval: dict,
    b_gyro: np.ndarray,
    gyro_correction: np.ndarray | None = None,
) -> np.ndarray:
    """
    Integrates bias-corrected 1000 Hz gyroscope measurements across [t_{k-1}, t_k]
    using midpoint/trapezoidal integration to return R_{I_{k-1}, I_k}.

    Gyro measurements are corrected with an optional fixed scale/misalignment
    matrix C_g: omega = C_g @ (omega_raw - b_g).
    """
    dts = interval["dt"]
    gyros = interval["gyro"]
    Cg = np.eye(3, dtype=np.float64) if gyro_correction is None else np.asarray(gyro_correction, dtype=np.float64)
    R_prev_curr = np.eye(3, dtype=np.float64)

    for j in range(len(dts)):
        omega_mid = Cg @ (0.5 * (gyros[j] + gyros[j + 1]) - b_gyro)
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
