"""Absolute Trajectory Error (ATE) for an estimated trajectory vs. a TUM ground truth.

The estimate is a ``*_trajectory.npz`` produced by ``trajectory.py``. By default
the newest ``*_tcp_trajectory.npz`` in ``logs/estimate/`` is used and its
``T_world_tcp`` poses are compared against ``groundtruth/traj_tum.txt``.

Translation uses the usual Umeyama similarity alignment. Rotation uses a
separate orientation-only Procrustes fit: the rotation that aligns a point
cloud does not generally align camera orientations, because VIO world frames
have an arbitrary gauge.

The ground-truth file is expected to be TUM RGB-D format:

    timestamp tx ty tz qx qy qz qw        # quaternion is x, y, z, w

Usage (from the repo root, with `uv`):

    uv run python ATE.py
    uv run python ATE.py --est logs/estimate/VID_20261117_145759_00_025_tcp_trajectory.npz
    uv run python ATE.py --time-offset -5.5
    uv run python ATE.py --pose-key T_world_cam1 --time-offset 0.0 --max-diff 0.02
    uv run python ATE.py --no-align          # raw error, no Umeyama alignment
    uv run python ATE.py --scale             # allow sim(3) scale correction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_GT = Path("groundtruth/traj_tum.txt")
EST_DIR = Path("logs/estimate")
POSE_KEYS = ("T_world_tcp", "T_world_cam1", "T_tcp0_tcp")


def latest_estimate() -> Path:
    """Return the newest TCP trajectory, or the newest plain trajectory."""
    candidates = list(EST_DIR.glob("*_tcp_trajectory.npz"))
    if not candidates:
        candidates = list(EST_DIR.glob("*_trajectory.npz"))
    if not candidates:
        raise FileNotFoundError(f"No *_trajectory.npz found in {EST_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion arrays [x, y, z, w] to (N, 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Ground truth contains a zero-length quaternion")
    q = q / norms
    x, y, z, w = q.T
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = np.empty((len(q), 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    rot[:, 0, 1] = 2.0 * (xy - wz)
    rot[:, 0, 2] = 2.0 * (xz + wy)
    rot[:, 1, 0] = 2.0 * (xy + wz)
    rot[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    rot[:, 1, 2] = 2.0 * (yz - wx)
    rot[:, 2, 0] = 2.0 * (xz - wy)
    rot[:, 2, 1] = 2.0 * (yz + wx)
    rot[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return rot


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a TUM trajectory file and return (timestamps, 4x4 poses)."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"{path} must have 8 columns: t tx ty tz qx qy qz qw")
    if len(data) == 0:
        raise ValueError(f"{path} is empty")

    timestamps = data[:, 0]
    poses = np.tile(np.eye(4, dtype=np.float64), (len(data), 1, 1))
    poses[:, :3, 3] = data[:, 1:4]
    poses[:, :3, :3] = quat_to_rot(data[:, 4:8])
    return timestamps, poses


def load_estimate(path: Path, pose_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamps and poses from an estimate .npz."""
    with np.load(path, allow_pickle=False) as data:
        if pose_key not in data.files:
            raise KeyError(f"{path} has no '{pose_key}'. Available: {data.files}")
        poses = np.array(data[pose_key], dtype=np.float64, copy=True)

        time_keys = [k for k in ("t_tcp", "t_cam1", "t") if k in data.files]
        if not time_keys:
            raise KeyError(f"{path} has no timestamp array (t_tcp/t_cam1/t)")
        timestamps = np.array(data[time_keys[0]], dtype=np.float64, copy=True)

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{pose_key} must have shape (N, 4, 4), got {poses.shape}")
    if len(poses) != len(timestamps):
        raise ValueError(
            f"timestamps ({len(timestamps)}) and poses ({len(poses)}) differ in length"
        )

    order = np.argsort(timestamps)
    return timestamps[order], poses[order]


def associate(
    gt_t: np.ndarray,
    est_t: np.ndarray,
    max_diff: float,
    time_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour timestamp association. Returns (gt_idx, est_idx, diffs)."""
    shifted = est_t + float(time_offset)
    insert = np.searchsorted(shifted, gt_t)
    left = np.clip(insert - 1, 0, len(shifted) - 1)
    right = np.clip(insert, 0, len(shifted) - 1)
    d_left = np.abs(shifted[left] - gt_t)
    d_right = np.abs(shifted[right] - gt_t)
    use_left = d_left <= d_right
    est_idx = np.where(use_left, left, right)
    diffs = np.minimum(d_left, d_right)
    valid = diffs <= float(max_diff)
    return np.nonzero(valid)[0], est_idx[valid], diffs[valid]


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    allow_scale: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Rigid (or similarity) alignment src -> dst using Umeyama's method."""
    if len(src) < 3:
        raise ValueError("Need at least 3 matched poses for alignment")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    u, _, vt = np.linalg.svd(src_c.T @ dst_c)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T

    scale = 1.0
    if allow_scale:
        denom = float(np.sum(src_c * src_c))
        if denom < 1e-12:
            raise ValueError("Cannot estimate scale from a stationary trajectory")
        scale = float(np.sum((src_c @ rot.T) * dst_c) / denom)
        if scale <= 0.0:
            raise ValueError("Estimated scale is non-positive")

    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def rotation_procrustes(
    est_rots: np.ndarray,
    gt_rots: np.ndarray,
) -> np.ndarray:
    """Find R such that R @ R_est ≈ R_gt for all matched poses."""
    est_rots = np.asarray(est_rots, dtype=np.float64)
    gt_rots = np.asarray(gt_rots, dtype=np.float64)
    # Minimize sum ||R @ E_i - G_i||_F. With column-vector rotations, the
    # solution is R = U @ Vt for SVD(sum G_i @ E_i.T).
    m = np.einsum("nij,nkj->ik", gt_rots, est_rots)
    u, _, vt = np.linalg.svd(m)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def apply_alignment(
    poses: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    """Apply a translation-alignment transform to 4x4 poses.

    The rotation comes from aligning point positions. It is applied to the
    rotation block for completeness, but rotation ATE is reported separately
    using ``rotation_procrustes()``.
    """
    aligned = poses.copy()
    aligned[:, :3, :3] = rot[None, :, :] @ poses[:, :3, :3]
    aligned[:, :3, 3] = scale * (poses[:, :3, 3] @ rot.T) + trans
    return aligned


def pose_errors(
    gt_poses: np.ndarray,
    est_poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation (m) and rotation (deg) errors per pose."""
    trans_err = np.linalg.norm(gt_poses[:, :3, 3] - est_poses[:, :3, 3], axis=1)

    rel = est_poses[:, :3, :3] @ np.transpose(gt_poses[:, :3, :3], (0, 2, 1))
    cos_angle = (np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0
    rot_err = np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return trans_err, rot_err


def report(label: str, err: np.ndarray, unit: str) -> float:
    rmse = float(np.sqrt(np.mean(err * err)))
    print(
        f"{label:<22} RMSE={rmse:9.5f} {unit} | "
        f"mean={np.mean(err):9.5f} | median={np.median(err):9.5f} | "
        f"std={np.std(err):9.5f} | max={np.max(err):9.5f}"
    )
    return rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--est", type=Path, default=None, help="Estimate trajectory .npz")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT, help="TUM ground truth .txt")
    parser.add_argument(
        "--pose-key",
        choices=POSE_KEYS,
        default="T_world_tcp",
        help="Pose array in the estimate .npz to evaluate (default: T_world_tcp)",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        default=0.02,
        help="Maximum timestamp difference for association in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Add this to estimate timestamps before matching (default: 0.0)",
    )
    parser.add_argument("--no-align", action="store_true", help="Skip Umeyama alignment")
    parser.add_argument(
        "--scale", action="store_true", help="Allow scale correction during alignment"
    )
    parser.add_argument(
        "--save-aligned",
        type=Path,
        default=None,
        help="Optional .npz path to save aligned estimates and matched ground truth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    est_path = args.est if args.est is not None else latest_estimate()

    gt_t, gt_poses = load_tum(args.gt)
    est_t, est_poses = load_estimate(est_path, args.pose_key)
    gt_idx, est_idx, diffs = associate(gt_t, est_t, args.max_diff, args.time_offset)

    if len(gt_idx) < 3:
        raise RuntimeError(
            "Fewer than 3 timestamp matches found. Check --max-diff and --time-offset."
        )

    gt_matched = gt_poses[gt_idx]
    est_matched = est_poses[est_idx]

    print(f"Estimate: {est_path} [{args.pose_key}]")
    print(f"Ground truth: {args.gt}")
    print(
        f"Matched {len(gt_idx)}/{len(gt_t)} ground-truth poses "
        f"(max timestamp diff {diffs.max():.4f} s, mean {diffs.mean():.4f} s)"
    )

    raw_trans, raw_rot = pose_errors(gt_matched, est_matched)
    report("Un-aligned translation", raw_trans, "m")
    report("Un-aligned rotation", raw_rot, "deg")

    rotation_aligned = est_matched.copy()
    if args.no_align:
        aligned = est_matched
    else:
        scale, rot, trans = umeyama(
            est_matched[:, :3, 3], gt_matched[:, :3, 3], allow_scale=args.scale
        )
        aligned = apply_alignment(est_matched, scale, rot, trans)
        t_str = f"t=[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}] m"
        print(f"Position alignment: scale={scale:.6f} {t_str}")

        rot_align = rotation_procrustes(
            est_matched[:, :3, :3], gt_matched[:, :3, :3]
        )
        rotation_aligned[:, :3, :3] = rot_align[None, :, :] @ est_matched[:, :3, :3]
        print(f"Orientation alignment: R={np.array2string(rot_align, precision=4)}")

    trans_err, _ = pose_errors(gt_matched, aligned)
    _, rot_err = pose_errors(gt_matched, rotation_aligned)
    ate = report("ATE translation", trans_err, "m")
    are = report("ATE rotation", rot_err, "deg")
    print(
        f"ATE RMSE summary: {ate:.6f} m, {are:.6f} deg over "
        f"{len(gt_idx)} poses"
    )

    if args.save_aligned is not None:
        aligned_poses = aligned.copy()
        aligned_poses[:, :3, :3] = rotation_aligned[:, :3, :3]
        args.save_aligned.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_aligned,
            t_gt=gt_t[gt_idx],
            t_est=est_t[est_idx] + args.time_offset,
            T_gt=gt_matched,
            T_est_aligned=aligned_poses,
            trans_error_m=trans_err,
            rot_error_deg=rot_err,
        )
        print(f"Saved aligned poses to {args.save_aligned}")


if __name__ == "__main__":
    main()
