"""Shared YAML configuration for rear-camera VIO runs."""

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = "config/vio.yaml"

DEFAULT_RUN_CONFIG = {
    "anchor": {
        "id": 0,
        "size_m": 0.100,
    },
    "frontend": {
        "max_corners_per_cam": 220,
        "min_track_len": 5,
    },
    "optimizer": {
        "max_iters": 40,
        "anchor_sigma_px": 0.05,
        "klt_sigma_px": 0.40,
        "anchor_huber_px": 10.0,
        "klt_huber_px": 2.5,
        "init_prior_rot": 0.0,
        "init_prior_trans": 0.0,
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
        elif isinstance(expected, int):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"{prefix}.{key} must be an integer, got {type(value).__name__}"
                )
        elif expected is None:
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{prefix}.{key} must be a string or null, got "
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
