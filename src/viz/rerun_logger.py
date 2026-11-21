"""Rerun logging of the optimized rear-camera trajectory..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

from pathlib import Path

import numpy as np
import rerun as rr


def log_optimized_vio_to_rerun(opt_res: dict, out_rrd: str):
    """Logs the optimized rear-camera trajectory and per-frame poses to Rerun."""
    Path(out_rrd).parent.mkdir(parents=True, exist_ok=True)
    rr.init("vio_batch_optimized", spawn=False)
    rr.save(out_rrd)

    try:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    except Exception:
        pass

    t = np.asarray(opt_res["t_cam1"], dtype=np.float64)
    T_wc = np.asarray(opt_res["T_world_cam1"], dtype=np.float64)
    rear_pts = T_wc[:, :3, 3].astype(np.float32)

    rr.log(
        "world/cam1_rear_trajectory",
        rr.LineStrips3D([rear_pts], radii=0.003),
        static=True,
    )

    # Per-frame pose visualization for inspection (RGB = XYZ axes).
    axis_len = float(np.max(np.linalg.norm(np.diff(rear_pts, axis=0), axis=1), initial=0.0))
    axis_len = float(np.clip(axis_len * 2.0, 0.01, 0.08))
    axis_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    for i in range(len(T_wc)):
        if not np.all(np.isfinite(T_wc[i])):
            continue
        p = T_wc[i, :3, 3].astype(np.float32)
        R = T_wc[i, :3, :3].astype(np.float32)
        axes = (R @ (np.eye(3, dtype=np.float32) * axis_len)).T
        rr.set_time("time", duration=float(t[i]) if i < len(t) and np.isfinite(t[i]) else float(i))
        rr.log(
            "world/cam1_pose/axes",
            rr.Arrows3D(
                origins=np.repeat(p[None, :], 3, axis=0),
                vectors=axes,
                colors=axis_colors,
            ),
        )
        rr.log(
            "world/cam1_pose/origin",
            rr.Points3D([p], radii=0.0025),
        )
