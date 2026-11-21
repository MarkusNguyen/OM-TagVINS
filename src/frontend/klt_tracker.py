"""IMU-aided fisheye KLT optical-flow tracker..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

import cv2
import numpy as np

from ..core.camera_model import project_kannala_brandt, unproject_kannala_brandt
from ..core.lie_algebra import so3_log
from .ransac import ransac_2pt_gyro_epipolar


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

            # Tighten forward-backward consistency during fast rotation, where
            # motion blur degrades KLT reliability.
            fb_thresh = 1.0
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
                rot_angle = float(np.linalg.norm(so3_log(dR_c1_c0)))
                fb_thresh = 1.0 / (1.0 + 2.0 * max(rot_angle, 0.0))
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
                & (fb_err < fb_thresh)
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
