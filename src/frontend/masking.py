"""Static foreground and fisheye-aperture masking..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import cv2
import numpy as np


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
