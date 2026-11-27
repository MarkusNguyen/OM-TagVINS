"""Shared YAML configuration for rear-camera VIO runs."""

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = "config/vio.yaml"

DEFAULT_RUN_CONFIG = {
    "anchor": {
        "id": 0,
        # Effective metric anchor size for this recording.
        "size_m": 0.180,
    },
    "frontend": {
        # Runtime-tuned defaults. config/vio.yaml is the effective source.
        "max_corners_per_cam": 220,
        "min_track_len": 5,
        "min_parallax_deg": 0.5,
        # Dead KLT tracks are still retired instead of growing one Python list
        # per observation forever. Specialist counts track span in keyframes;
        # this repo applies the same numeric cap to frame observations.
        "max_track_len": 300,
        "max_triangulation_obs": 8,
        # AprilTag detection on every full-resolution frame dominates runtime.
        # Detect on a downscaled image every N-th frame, then refine corners on
        # the full-resolution image.
        "tag_detect_stride": 1,
        "tag_detect_scale": 0.5,
    },
    "optimizer": {
        # Specialist SolverCfg.max_iterations.
        "max_iters": 30,
        # Batch-graph size controls. Initialization still uses every anchor
        # sighting; only the optimizer's factor set is subsampled.
        "anchor_factor_stride": 4,
        "max_klt_obs_per_landmark": 30,
        # Optimize on every N-th camera frame and interpolate the full-rate
        # output. Front-end detection and trajectory export remain full rate.
        "backend_frame_stride": 2,
        "anchor_sigma_px": 1.05,
        "klt_sigma_px": 0.5,
        "anchor_huber_px": 1.41,
        "klt_huber_px": 0.67,
        "init_prior_rot": 0.0,
        "init_prior_trans": 0.0,
        # Inflate KLT sigma/Huber by (1 + scale * ||omega|| [rad/s]) during
        # fast rotation to reduce motion-blur / high-rate coupling influence.
        "klt_omega_rate_scale": 0.0,
        # Rolling-shutter readout in seconds. None uses the value logged by
        # converter.py from the .insv telemetry; 0 disables compensation.
        "rolling_shutter_readout_s": None,
        # Online estimation of the camera-minus-IMU timing offset.
        "estimate_time_offset": False,
        "time_offset_prior_s": 0.005,
    },
    "output": {
        # Added to exported TCP timestamps so the documented ATE offset stays
        # -5.5 s even when the best trajectory shape aligns at a slightly
        # different raw timestamp offset.
        "time_origin_offset_s": 0.0,
    },
}


def _merge_known(base: dict, override: dict, prefix: str = "config") -> dict:
    """Recursively merge override into base, rejecting unknown or mistyped keys."""
    if not isinstance(override, dict):
        raise ValueError(f"{prefix} must be a mapping")

    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}.{key}")

        expected = base[key]
        if isinstance(expected, dict):
            _merge_known(expected, value, f"{prefix}.{key}")
            continue

        # YAML may parse an integer as an int for a float field; allow that.
        if isinstance(expected, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{prefix}.{key} must be a number, got {type(value).__name__}"
                )
            value = float(value)
        elif isinstance(expected, bool):
            if not isinstance(value, bool):
                raise ValueError(
                    f"{prefix}.{key} must be a boolean, got {type(value).__name__}"
                )
        elif isinstance(expected, int):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"{prefix}.{key} must be an integer, got {type(value).__name__}"
                )
        elif expected is None:
            if value is not None and not isinstance(value, (str, int, float)):
                raise ValueError(
                    f"{prefix}.{key} must be a string, number, or null, got "
                    f"{type(value).__name__}"
                )
        elif isinstance(expected, str):
            if not isinstance(value, str):
                raise ValueError(
                    f"{prefix}.{key} must be a string, got {type(value).__name__}"
                )

        base[key] = value

    return base


def load_run_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the VIO run configuration, falling back to built-in values."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"VIO configuration file not found: {path}")

    with path.open("r") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError(f"VIO configuration root must be a mapping: {path}")

    return _merge_known(deepcopy(DEFAULT_RUN_CONFIG), user_cfg)
