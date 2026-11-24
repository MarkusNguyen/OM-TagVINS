"""Rear-camera visual front-end pipeline entry point.

The implementation lives in focused modules under ``src/``; this file wires
them together and exposes the two public functions used by the rest of the
project.
"""

from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

from preprocess import prepare_vio_inputs
from config.settings import DEFAULT_CONFIG_PATH, load_run_config
from src.core.camera_model import project_kannala_brandt, unproject_kannala_brandt
from src.core.constants import ACTIVE_CAMS, CAMERA_IMAGE_KEYS, REAR_CAM
from src.core.lie_algebra import so3_exp
from src.frontend.apriltag import (
    get_tag_corners_3d,
    refine_tag_corners_virtual_pinhole,
    solve_tag_pose_fisheye,
)
from src.frontend.gyro import (
    compensate_rolling_shutter_rays,
    compute_camera_relative_rotation,
    integrate_gyro_rotation,
)
from src.frontend.klt_tracker import IMUAidedFisheyeKLTTracker
from src.frontend.masking import build_static_masks
from src.frontend.triangulation import triangulate_klt_tracks

def extract_visual_measurements(
    vio_data: dict,
    anchor_tags: dict,
    max_corners_per_cam: int = 220,
    min_track_len: int = 5,
    min_parallax_deg: float = 0.15,
) -> dict:
    """
    Runs stationary anchor AprilTag detection and IMU-aided fisheye KLT tracking
    on the rear camera across all synchronized frames.
    """
    cams = vio_data["cams"]
    cam1 = cams[REAR_CAM]
    b_gyro_0 = vio_data["b_gyro_0"]
    imu_intervals_cam1 = vio_data["imu_intervals_cam1"]
    gyro_correction = vio_data.get("imu_cfg", {}).get("gyro_correction")

    detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=1)

    trackers = {
        cam_name: IMUAidedFisheyeKLTTracker(
            cams[cam_name],
            build_static_masks(cams[cam_name], cam_name),
            max_corners_per_cam,
        )
        for cam_name in ACTIVE_CAMS
    }

    anchor_corners_world = {}
    anchor_corners_local = {}
    for tid, info in anchor_tags.items():
        local_4x3 = get_tag_corners_3d(info["size_m"])
        local_hom = np.hstack([local_4x3, np.ones((4, 1), dtype=np.float64)])
        anchor_corners_local[tid] = local_4x3
        anchor_corners_world[tid] = (info["T_world_tag"] @ local_hom.T).T[:, :3]

    anchor_obs = []
    init_poses_world_imu = {}
    last_tag_poses = {}

    num_frames = len(vio_data["rear_imgs"])
    for k in range(num_frames):
        for cam_name in ACTIVE_CAMS:
            bgr = vio_data[CAMERA_IMAGE_KEYS[cam_name]][k]
            cam = cams[cam_name]

            # Integrate 1000 Hz gyro between frame k-1 and k on the rear
            # camera's timeshifted timeline.
            R_I_prev_curr = (
                integrate_gyro_rotation(
                    imu_intervals_cam1[k - 1], b_gyro_0, gyro_correction
                )
                if k > 0
                else None
            )

            if k < len(imu_intervals_cam1):
                omega_node = imu_intervals_cam1[k]["gyro"][0]
            else:
                omega_node = imu_intervals_cam1[-1]["gyro"][-1]
            if gyro_correction is None:
                omega_unb = omega_node - b_gyro_0
            else:
                omega_unb = gyro_correction @ (omega_node - b_gyro_0)

            gray_rear = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            dets = detector.detect(gray_rear)

            for det in dets:
                tid = int(det.tag_id)
                if tid not in anchor_tags:
                    continue

                corners_uv = refine_tag_corners_virtual_pinhole(
                    gray_rear, np.asarray(det.corners, dtype=np.float64), det.tag_id, cam1, detector
                )
                info = anchor_tags[tid]
                rays = unproject_kannala_brandt(corners_uv, cam)
                anchor_obs.append({
                    "frame_idx": k,
                    "cam_name": cam_name,
                    "tag_id": tid,
                    "corners_uv": corners_uv,
                    "corners_rays": rays,
                    "corners_world": anchor_corners_world[tid],
                    # Local tag-frame corners let the backend optimize the anchor
                    # pose instead of treating it as a fixed world transform.
                    "corners_local": anchor_corners_local[tid],
                })

                # PnP initialization should use the same rolling-shutter model
                # as the solver. The stored anchor observations stay in raw
                # pixel coordinates so the optimizer applies only once.
                rays_nominal = compensate_rolling_shutter_rays(
                    corners_uv, rays, cam, omega_unb
                )
                uv_nominal = project_kannala_brandt(rays_nominal, cam)
                T_cam_tag = solve_tag_pose_fisheye(
                    uv_nominal, cam, info["size_m"], prev_T_cam_tag=last_tag_poses.get(tid)
                )
                if T_cam_tag is not None:
                    last_tag_poses[tid] = T_cam_tag
                    if k not in init_poses_world_imu:
                        T_world_imu = (
                            info["T_world_tag"]
                            @ np.linalg.inv(T_cam_tag)
                            @ cam["T_cam_imu"]
                        )
                        init_poses_world_imu[k] = T_world_imu

            dR_c1_c0 = compute_camera_relative_rotation(R_I_prev_curr, cam) if R_I_prev_curr is not None else None
            trackers[cam_name].step(k, gray_rear, dR_c1_c0=dR_c1_c0)

    if not anchor_obs:
        raise RuntimeError(
            "No AprilTag anchor observations were found. The VIO backend "
            "needs at least one anchor sighting to fix the world gauge."
        )
    if not init_poses_world_imu:
        raise RuntimeError(
            "The AprilTag anchor was detected but planar PnP failed for every "
            "sighting; cannot initialize VIO poses."
        )

    klt_tracks = {
        cam_name: {tid: obs for tid, obs in trk.tracks.items() if len(obs) >= min_track_len}
        for cam_name, trk in trackers.items()
    }

    landmarks_world = triangulate_klt_tracks(
        klt_tracks,
        init_poses_world_imu,
        cams,
        min_parallax_deg=min_parallax_deg,
    )

    return {
        "anchor_obs": anchor_obs,
        "init_poses_world_imu": init_poses_world_imu,
        "klt_tracks": klt_tracks,
        "landmarks_world": landmarks_world,
        # Initial guesses for the optimizable anchor pose variables. The backend
        # keeps these as free variables with a tilt-free gauge prior.
        "anchor_poses_world": {
            tid: np.asarray(info["T_world_tag"], dtype=np.float64).copy()
            for tid, info in anchor_tags.items()
        },
    }


def run_frontend(
    rrd: str | Path,
    calib: str | Path = "config/cam-imu.yaml",
    config: str | Path = DEFAULT_CONFIG_PATH,
    *,
    anchor_id: int | None = None,
    anchor_size: float | None = None,
    max_corners_per_cam: int | None = None,
    min_track_len: int | None = None,
    min_parallax_deg: float | None = None,
) -> dict:
    """Run the rear-camera visual front-end on a base .rrd recording.

    Parameters left as None fall back to the VIO YAML configuration.  Returns
    the visual-measurements dictionary.
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

    print(f"VIO config: {config}")

    vio_data = prepare_vio_inputs(rrd, calib)

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

    print(f"Frames processed:        {len(vio_data['t_cam1'])}")
    print(f"Anchor observations:     {len(visual_meas['anchor_obs'])}")
    print(f"Initial PnP poses:       {len(visual_meas['init_poses_world_imu'])}")
    print(
        f"Triangulated KLT (rear): "
        f"{len(visual_meas['landmarks_world'][REAR_CAM])}"
    )

    return visual_meas
