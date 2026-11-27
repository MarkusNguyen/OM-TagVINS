#!/usr/bin/env python3
# Usage:
#   python trajectory.py --rrd logs/base/<input>.rrd
#   python trajectory.py
#   python trajectory.py --npz logs/estimate/<stem>_trajectory.npz
#
# Output:
#   logs/estimate/<stem>_tcp_trajectory.npz
#   logs/estimate/<stem>_tcp_vio.rrd
#
# trajectory.py calls the origin optimize.py for VIO.  The .npz contains TCP
# poses.  The .rrd shows a static TCP line plus a moving TCP marker at
# world/tcp_marker.

import argparse
import tempfile
from pathlib import Path

import numpy as np
import yaml

# The VIO YAML is the origin config.
from config.settings import DEFAULT_CONFIG_PATH, load_run_config


ROOT = Path(__file__).resolve().parent
OPTIMIZE_SCRIPT = ROOT / "optimize.py"

DEFAULT_CONFIG = "config/cam-in-tcp.yaml"
DEFAULT_CALIB = "config/cam-imu.yaml"
DEFAULT_ESTIMATE_DIR = Path("logs/estimate")


def validate_transform(matrix, name: str) -> np.ndarray:
    """Validate a 4x4 homogeneous transform."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation block is not orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(f"{name} rotation block has a non-positive determinant")

    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) <= 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    transform[:3, :3] = rotation
    return transform


def load_camera_tcp_transform(config_path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (T_cam_tcp, T_tcp_cam, source_description)."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera-to-TCP configuration not found: {path}")

    text = path.read_text().strip()
    if not text:
        identity = np.eye(4, dtype=np.float64)
        return identity, identity.copy(), "identity (empty config)"

    raw = yaml.safe_load(text) or {}
    if isinstance(raw, list):
        key = "T_tcp_cam"
        matrix = raw
    elif isinstance(raw, dict):
        if "T_tcp_cam" in raw:
            key = "T_tcp_cam"
            matrix = raw["T_tcp_cam"]
        elif "T_cam_tcp" in raw:
            key = "T_cam_tcp"
            matrix = raw["T_cam_tcp"]
        else:
            raise ValueError(
                f"{path} must define T_tcp_cam or T_cam_tcp (or be empty for identity)"
            )
    else:
        raise ValueError(f"{path} must contain a mapping or a 4x4 list")

    transform = validate_transform(matrix, key)
    if key == "T_tcp_cam":
        T_tcp_cam = transform
        T_cam_tcp = np.linalg.inv(transform)
        source = "T_tcp_cam (camera frame -> TCP frame)"
    else:
        T_cam_tcp = transform
        T_tcp_cam = np.linalg.inv(transform)
        source = "T_cam_tcp (TCP frame -> camera frame)"
    return T_cam_tcp, T_tcp_cam, source


def load_trajectory(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load camera poses and timestamps from a trajectory .npz file."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "T_world_cam1" not in data:
            raise KeyError(f"{path} does not contain T_world_cam1")
        camera_poses = np.array(data["T_world_cam1"], dtype=np.float64, copy=True)
        if "t_cam1" in data:
            timestamps = np.array(data["t_cam1"], dtype=np.float64, copy=True)
        else:
            timestamps = np.arange(len(camera_poses), dtype=np.float64)

    if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
        raise ValueError(
            f"T_world_cam1 must have shape (N, 4, 4), got {camera_poses.shape}"
        )
    if len(camera_poses) == 0:
        raise ValueError("T_world_cam1 is empty")
    if len(timestamps) != len(camera_poses):
        raise ValueError(
            f"t_cam1 length {len(timestamps)} does not match poses {len(camera_poses)}"
        )
    if not np.isfinite(camera_poses).all():
        raise ValueError("T_world_cam1 contains non-finite values")

    return camera_poses, timestamps


def latest_trajectory(estimate_dir: str | Path) -> Path:
    """Return the newest non-TCP trajectory .npz in the estimate directory."""
    directory = Path(estimate_dir)
    candidates = [
        path
        for path in directory.glob("*_trajectory.npz")
        if not path.name.endswith("_tcp_trajectory.npz")
    ]
    if not candidates:
        raise FileNotFoundError(f"No trajectory .npz files found in {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def camera_to_tcp(camera_poses: np.ndarray, T_cam_tcp: np.ndarray) -> np.ndarray:
    """Convert world-frame camera poses to world-frame TCP poses."""
    return np.matmul(camera_poses, T_cam_tcp)


def rebase_trajectory(tcp_poses: np.ndarray) -> np.ndarray:
    """Express every TCP pose relative to the first TCP pose."""
    reference_inverse = np.linalg.inv(tcp_poses[0])
    relative_poses = np.matmul(reference_inverse, tcp_poses)
    relative_poses[0] = np.eye(4, dtype=np.float64)
    return relative_poses


def log_tcp_trajectory(
    out_rrd: str | Path,
    relative_poses: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> None:
    """Log the TCP trajectory and per-frame poses to Rerun."""
    import rerun as rr

    path = Path(out_rrd)
    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("tcp_trajectory", spawn=False)
    rr.save(path)

    radius = 0.0005
    poses = np.asarray(relative_poses, dtype=np.float64)
    positions = poses[:, :3, 3].astype(np.float32)
    try:
        try:
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        except Exception:
            pass

        rr.log(
            "world/tcp_trajectory",
            rr.LineStrips3D([positions], radii=radius),
            static=True,
        )

        axis_len = float(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1), initial=0.0))
        axis_len = float(np.clip(axis_len * 2.0, 0.005, 0.05))
        axis_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        if timestamps is not None:
            t_tcp = np.asarray(timestamps, dtype=np.float64)
            if len(t_tcp) != len(poses):
                raise ValueError(
                    f"timestamps length {len(t_tcp)} does not match poses {len(poses)}"
                )
        else:
            t_tcp = np.arange(len(poses), dtype=np.float64)

        valid = np.isfinite(t_tcp) & np.isfinite(positions).all(axis=1) & np.isfinite(poses).all(axis=(1, 2))
        for i, (t, pose) in enumerate(zip(t_tcp[valid], poses[valid])):
            rr.set_time("time", duration=float(t))
            p = pose[:3, 3].astype(np.float32)
            R = pose[:3, :3].astype(np.float32)
            axes = (R @ (np.eye(3, dtype=np.float32) * axis_len)).T
            rr.log(
                "world/tcp_pose/axes",
                rr.Arrows3D(
                    origins=np.repeat(p[None, :], 3, axis=0),
                    vectors=axes,
                    colors=axis_colors,
                ),
            )
            rr.log(
                "world/tcp_marker",
                rr.Points3D(
                    [p],
                    radii=[radius],
                    labels=[f"{float(t):.2f}s"],
                ),
            )
    finally:
        rr.disconnect()


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    """Derive TCP output paths from an input trajectory or recording name."""
    stem = input_path.stem
    for suffix in ("_trajectory", "_tcp_trajectory", "_tcp_vio", "_vio"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out_npz = DEFAULT_ESTIMATE_DIR / f"{stem}_tcp_trajectory.npz"
    out_rrd = DEFAULT_ESTIMATE_DIR / f"{stem}_tcp_vio.rrd"
    return out_npz, out_rrd


def run_vio_via_optimize(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Run the optimize.py VIO pipeline and load its camera trajectory."""
    from optimize import run_vio

    if not OPTIMIZE_SCRIPT.is_file():
        raise FileNotFoundError(f"VIO backend not found: {OPTIMIZE_SCRIPT}")

    tmp_root = Path(tempfile.gettempdir()) / "opencode"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="trajectory_vio_", dir=tmp_root) as tmp:
        vio_npz = Path(tmp) / "vio_trajectory.npz"
        print(f"VIO backend: {OPTIMIZE_SCRIPT}", flush=True)
        print(f"VIO config: {args.vio_config}", flush=True)
        print(f"Input recording: {args.rrd}", flush=True)
        run_vio(
            rrd=args.rrd,
            calib=args.calib,
            config=args.vio_config,
            out_npz=vio_npz,
            viz="",
            anchor_id=args.anchor_id,
            anchor_size=args.anchor_size,
            max_corners_per_cam=args.max_corners_per_cam,
            min_track_len=args.min_track_len,
            min_parallax_deg=args.min_parallax_deg,
            max_track_len=args.max_track_len,
            max_triangulation_obs=args.max_triangulation_obs,
            tag_detect_stride=args.tag_detect_stride,
            tag_detect_scale=args.tag_detect_scale,
            max_iters=args.max_iters,
            anchor_sigma=args.anchor_sigma,
            klt_sigma=args.klt_sigma,
            anchor_huber=args.anchor_huber,
            klt_huber=args.klt_huber,
            init_prior_rot=args.init_prior_rot,
            init_prior_trans=args.init_prior_trans,
            klt_omega_rate_scale=args.klt_omega_rate_scale,
            rolling_shutter_readout_s=args.rolling_shutter_readout_s,
            estimate_time_offset=args.estimate_time_offset,
            time_offset_prior_s=args.time_offset_prior_s,
            anchor_factor_stride=args.anchor_factor_stride,
            max_klt_obs_per_landmark=args.max_klt_obs_per_landmark,
            backend_frame_stride=args.backend_frame_stride,
        )
        return load_trajectory(vio_npz)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--vio-config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_run_config(config_args.vio_config)

    parser = argparse.ArgumentParser(
        description="Run VIO with optimize.py and write a TCP trajectory"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rrd",
        default=None,
        help="Input raw base .rrd; runs optimize.py first",
    )
    mode.add_argument(
        "--npz",
        default=None,
        help="Input existing camera trajectory .npz; skips optimize.py",
    )

    parser.add_argument(
        "--vio-config",
        default=config_args.vio_config,
        help=f"VIO YAML configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--calib",
        default=DEFAULT_CALIB,
        help=f"Camera/IMU calibration YAML (default: {DEFAULT_CALIB})",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Camera-to-TCP extrinsics (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--out-npz",
        default=None,
        help="Output TCP trajectory .npz (default: <stem>_tcp_trajectory.npz)",
    )
    parser.add_argument(
        "--viz",
        default=None,
        help="Output TCP Rerun .rrd (default: <stem>_tcp_vio.rrd)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Do not write a Rerun recording",
    )

    vio = parser.add_argument_group("VIO options (forwarded to optimize.py)")
    vio.add_argument(
        "--anchor-id",
        type=int,
        default=cfg["anchor"]["id"],
        help="Stationary anchor AprilTag ID",
    )
    vio.add_argument(
        "--anchor-size",
        type=float,
        default=cfg["anchor"]["size_m"],
        help="Anchor tag side length (m)",
    )
    vio.add_argument(
        "--max-corners-per-cam",
        type=int,
        default=cfg["frontend"]["max_corners_per_cam"],
        help="Maximum tracked KLT corners per camera",
    )
    vio.add_argument(
        "--min-track-len",
        type=int,
        default=cfg["frontend"]["min_track_len"],
        help="Minimum KLT track length in frames",
    )
    vio.add_argument(
        "--min-parallax-deg",
        type=float,
        default=cfg["frontend"]["min_parallax_deg"],
        help="Minimum KLT bearing parallax for triangulation (degrees)",
    )
    vio.add_argument(
        "--max-track-len",
        type=int,
        default=cfg["frontend"]["max_track_len"],
        help="Maximum observations retained per KLT track (runtime guard)",
    )
    vio.add_argument(
        "--max-triangulation-obs",
        type=int,
        default=cfg["frontend"]["max_triangulation_obs"],
        help="Maximum observations sampled per track during triangulation",
    )
    vio.add_argument(
        "--tag-detect-stride",
        type=int,
        default=cfg["frontend"]["tag_detect_stride"],
        help="Run full AprilTag detection every N-th frame",
    )
    vio.add_argument(
        "--tag-detect-scale",
        type=float,
        default=cfg["frontend"]["tag_detect_scale"],
        help="Scale factor for AprilTag detection image (corners are refined at full resolution)",
    )
    vio.add_argument(
        "--max-iters",
        type=int,
        default=cfg["optimizer"]["max_iters"],
        help="Maximum LM iterations",
    )
    vio.add_argument(
        "--anchor-sigma",
        type=float,
        default=cfg["optimizer"]["anchor_sigma_px"],
        help="Anchor corner reprojection sigma (px)",
    )
    vio.add_argument(
        "--klt-sigma",
        type=float,
        default=cfg["optimizer"]["klt_sigma_px"],
        help="KLT reprojection noise sigma (px)",
    )
    vio.add_argument(
        "--anchor-huber",
        type=float,
        default=cfg["optimizer"]["anchor_huber_px"],
        help="Anchor Huber delta (px)",
    )
    vio.add_argument(
        "--klt-huber",
        type=float,
        default=cfg["optimizer"]["klt_huber_px"],
        help="KLT Huber delta (px)",
    )
    vio.add_argument(
        "--klt-omega-rate-scale",
        type=float,
        default=cfg["optimizer"]["klt_omega_rate_scale"],
        help="Inflate KLT sigma/Huber by (1 + scale * ||omega|| [rad/s])",
    )
    vio.add_argument(
        "--init-prior-rot",
        type=float,
        default=cfg["optimizer"]["init_prior_rot"],
        help="PnP branch-continuity rotation prior sigma (rad); 0 disables",
    )
    vio.add_argument(
        "--init-prior-trans",
        type=float,
        default=cfg["optimizer"]["init_prior_trans"],
        help="PnP branch-continuity translation prior sigma (m); 0 disables",
    )
    vio.add_argument(
        "--rolling-shutter-readout-s",
        type=float,
        default=cfg["optimizer"]["rolling_shutter_readout_s"],
        help=(
            "Rolling-shutter full-frame readout in seconds. Default null uses "
            "the .insv metadata logged by converter.py; 0 disables."
        ),
    )
    timing = vio.add_mutually_exclusive_group()
    timing.add_argument(
        "--estimate-time-offset",
        dest="estimate_time_offset",
        action="store_true",
        default=cfg["optimizer"]["estimate_time_offset"],
        help="Estimate the camera-minus-IMU time offset (default: from VIO config)",
    )
    timing.add_argument(
        "--no-estimate-time-offset",
        dest="estimate_time_offset",
        action="store_false",
        help="Fix the camera-minus-IMU time offset at zero",
    )
    vio.add_argument(
        "--time-offset-prior-s",
        type=float,
        default=cfg["optimizer"]["time_offset_prior_s"],
        help="Prior sigma on the estimated camera-minus-IMU time offset (s)",
    )
    vio.add_argument(
        "--anchor-factor-stride",
        type=int,
        default=cfg["optimizer"]["anchor_factor_stride"],
        help="Add anchor gauge factors only every N-th detection frame",
    )
    vio.add_argument(
        "--max-klt-obs-per-landmark",
        type=int,
        default=cfg["optimizer"]["max_klt_obs_per_landmark"],
        help="Maximum KLT observations per landmark added to the batch graph",
    )
    vio.add_argument(
        "--backend-frame-stride",
        type=int,
        default=cfg["optimizer"]["backend_frame_stride"],
        help=(
            "Optimize every N-th camera frame and interpolate the trajectory "
            "output (keep at 2; larger values can destabilize LM)"
        ),
    )

    output_args = parser.add_argument_group("Output options")
    output_args.add_argument(
        "--time-origin-offset",
        type=float,
        default=cfg["output"]["time_origin_offset_s"],
        help="Seconds added to exported TCP timestamps before evaluation",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rrd:
        input_path = Path(args.rrd)
        camera_poses, timestamps = run_vio_via_optimize(args)
        input_label = "Input recording"
        mode_label = "optimize.py VIO + TCP"
    else:
        input_path = Path(args.npz) if args.npz else latest_trajectory(DEFAULT_ESTIMATE_DIR)
        camera_poses, timestamps = load_trajectory(input_path)
        input_label = "Input trajectory"
        mode_label = "TCP transform only"

    output_npz, default_rrd = default_output_paths(input_path)
    if args.out_npz:
        output_npz = Path(args.out_npz)
    if args.no_viz or args.viz == "":
        output_rrd = None
    else:
        output_rrd = Path(args.viz or default_rrd)

    T_cam_tcp, T_tcp_cam, source = load_camera_tcp_transform(args.config)

    tcp_poses = camera_to_tcp(camera_poses, T_cam_tcp)
    relative_poses = rebase_trajectory(tcp_poses)
    export_timestamps = np.asarray(timestamps, dtype=np.float64) + float(
        args.time_origin_offset
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_arrays = {
        "t_tcp": export_timestamps,
        "T_world_tcp": tcp_poses,
        "T_tcp0_tcp": relative_poses,
        "T_world_cam1": camera_poses,
        "T_cam_tcp": T_cam_tcp,
        "T_tcp_cam": T_tcp_cam,
    }
    np.savez(output_npz, **output_arrays)

    print(f"Mode: {mode_label}")
    print(f"{input_label}: {input_path}")
    print(f"Extrinsics: {args.config} ({source})")
    if float(args.time_origin_offset) != 0.0:
        print(f"Time-origin offset applied to export: {args.time_origin_offset:+.3f} s")
    if np.allclose(T_cam_tcp, np.eye(4), atol=1e-12):
        print("Note: camera-to-TCP transform is identity (placeholder)")
    print(f"TCP poses: {len(tcp_poses)}")
    print(f"First world TCP position: {tcp_poses[0, :3, 3]}")
    print(f"Rebased first position: {relative_poses[0, :3, 3]}")
    print(f"Rebased last position:  {relative_poses[-1, :3, 3]}")
    print(f"Saved TCP trajectory to {output_npz}")

    if output_rrd is not None:
        log_tcp_trajectory(output_rrd, relative_poses, export_timestamps)
        print(f"Saved TCP Rerun recording to {output_rrd}")
        print("Moving TCP marker logged as world/tcp_marker on the time timeline")


if __name__ == "__main__":
    main()
