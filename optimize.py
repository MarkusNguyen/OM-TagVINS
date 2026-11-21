"""Full-batch IMU-aided VIO pipeline entry point.

The implementation lives in focused modules under ``src/``; this file wires
them together and exposes ``run_vio``.  The imported helper names are kept in
scope as a convenience for callers that previously imported them from here.
"""

from pathlib import Path

import numpy as np

from frontend import extract_visual_measurements
from preprocess import prepare_vio_inputs
from config.settings import DEFAULT_CONFIG_PATH, load_run_config
from src.backend.imu_preintegration import PreintegratedIMUFactor
from src.backend.vio_initialization import (
    extract_frame_gyros,
    initialize_vio_states,
    pack_visual_observations,
)
from src.backend.vio_solver import (
    evaluate_visual_residuals_and_jacobians,
    huber_irls_weights,
    run_batch_vio_optimization,
)
from src.core.camera_model import project_kannala_brandt, project_kb_with_jacobian
from src.core.lie_algebra import (
    skew,
    so3_exp,
    so3_inv_right_jacobian,
    so3_log,
    so3_right_jacobian,
)
from src.viz.rerun_logger import log_optimized_vio_to_rerun

def run_vio(
    rrd: str | Path,
    calib: str | Path = "config/cam-imu.yaml",
    config: str | Path = DEFAULT_CONFIG_PATH,
    out_npz: str | Path | None = None,
    viz: str | Path | None = None,
    *,
    anchor_id: int | None = None,
    anchor_size: float | None = None,
    max_corners_per_cam: int | None = None,
    min_track_len: int | None = None,
    min_parallax_deg: float | None = None,
    max_iters: int | None = None,
    anchor_sigma: float | None = None,
    klt_sigma: float | None = None,
    anchor_huber: float | None = None,
    klt_huber: float | None = None,
    init_prior_rot: float | None = None,
    init_prior_trans: float | None = None,
    klt_omega_rate_scale: float | None = None,
    rolling_shutter_readout_s: float | None = None,
    estimate_time_offset: bool | None = None,
    time_offset_prior_s: float | None = None,
) -> dict:
    """Run full-batch IMU-aided VIO on a base .rrd recording.

    Parameters left as None fall back to the VIO YAML configuration.  Pass
    viz="" to skip Rerun output.  Returns the optimization result dictionary.
    """
    cfg = load_run_config(config)

    if anchor_id is None:
        anchor_id = cfg["anchor"]["id"]
    if anchor_size is None:
        anchor_size = cfg["anchor"]["size_m"]
    if max_corners_per_cam is None:
        max_corners_per_cam = cfg["frontend"]["max_corners_per_cam"]
    if min_track_len is None:
        min_track_len = cfg["frontend"]["min_track_len"]
    if min_parallax_deg is None:
        min_parallax_deg = cfg["frontend"]["min_parallax_deg"]
    if max_iters is None:
        max_iters = cfg["optimizer"]["max_iters"]
    if anchor_sigma is None:
        anchor_sigma = cfg["optimizer"]["anchor_sigma_px"]
    if klt_sigma is None:
        klt_sigma = cfg["optimizer"]["klt_sigma_px"]
    if anchor_huber is None:
        anchor_huber = cfg["optimizer"]["anchor_huber_px"]
    if klt_huber is None:
        klt_huber = cfg["optimizer"]["klt_huber_px"]
    if init_prior_rot is None:
        init_prior_rot = cfg["optimizer"]["init_prior_rot"]
    if init_prior_trans is None:
        init_prior_trans = cfg["optimizer"]["init_prior_trans"]
    if klt_omega_rate_scale is None:
        klt_omega_rate_scale = cfg["optimizer"]["klt_omega_rate_scale"]
    if rolling_shutter_readout_s is None:
        rolling_shutter_readout_s = cfg["optimizer"]["rolling_shutter_readout_s"]
    if estimate_time_offset is None:
        estimate_time_offset = cfg["optimizer"]["estimate_time_offset"]
    if time_offset_prior_s is None:
        time_offset_prior_s = cfg["optimizer"]["time_offset_prior_s"]

    print(f"VIO config: {config}")

    # Derive distinctive output names from the input recording.
    rrd_stem = Path(rrd).stem
    if out_npz is None:
        out_npz = str(Path("logs/estimate") / f"{rrd_stem}_trajectory.npz")
    if viz is None:
        viz = str(Path("logs/estimate") / f"{rrd_stem}_vio.rrd")

    vio_data = prepare_vio_inputs(rrd, calib)

    # Explicit config/CLI readout overrides the metadata logged by converter.py.
    if rolling_shutter_readout_s is not None:
        rolling_shutter_readout_s = float(rolling_shutter_readout_s)
        for cam in vio_data["cams"].values():
            cam["rolling_shutter_readout_s"] = rolling_shutter_readout_s
        vio_data["rolling_shutter_readout_s"] = rolling_shutter_readout_s

    print(
        f"Rolling-shutter readout: "
        f"{vio_data['cams']['cam1'].get('rolling_shutter_readout_s')} s | "
        f"estimate time offset: {estimate_time_offset}"
    )

    anchor_tags = {
        anchor_id: {
            "size_m": anchor_size,
            "T_world_tag": np.eye(4, dtype=np.float64),
        }
    }

    visual_meas = extract_visual_measurements(
        vio_data,
        anchor_tags=anchor_tags,
        max_corners_per_cam=max_corners_per_cam,
        min_track_len=min_track_len,
        min_parallax_deg=min_parallax_deg,
    )

    opt_res = run_batch_vio_optimization(
        vio_data,
        visual_meas,
        max_iters=max_iters,
        anchor_sigma_px=anchor_sigma,
        klt_sigma_px=klt_sigma,
        huber_delta_px=klt_huber,
        anchor_huber_delta_px=anchor_huber,
        init_prior_rot=init_prior_rot,
        init_prior_trans=init_prior_trans,
        klt_omega_rate_scale=klt_omega_rate_scale,
        estimate_time_offset=bool(estimate_time_offset),
        time_offset_prior_s=float(time_offset_prior_s),
    )

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **opt_res)
    print(f"Saved smoothed 6-DoF trajectory to {out_npz}")

    if viz:
        log_optimized_vio_to_rerun(opt_res, out_rrd=viz)
        print(f"Saved rear-camera trajectory to {viz}")

    return opt_res
