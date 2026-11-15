# OM-TagVINS

Rear-camera-only visual-inertial odometry with a single stationary AprilTag anchor.

The pipeline estimates a metric 6-DoF camera trajectory by combining:

- 1000 Hz IMU preintegration
- AprilTag corner observations of one stationary `tag36h11` anchor
- Gyro-aided KLT feature tracking
- A full-batch on-manifold Levenberg-Marquardt optimizer

Only the rear camera (`cam1`) is used for VIO. The front camera is not used.

## Project layout

```text
.
├── converter.py          # Optional: convert Insta360 .insv to a Rerun .rrd
├── preprocess.py         # Load and synchronize camera/IMU data
├── frontend.py           # AprilTag detection, PnP, KLT tracking, triangulation
├── optimize.py           # Batch VIO optimizer and main CLI
├── settings.py           # YAML configuration loader
├── config/
│   ├── cam-imu.yaml      # Camera intrinsics, extrinsics, timeshifts, IMU noise
│   ├── vio.yaml          # Runtime VIO parameters
│   └── anchor-tags.yaml  # Single-tag anchor reference
├── input/                # Source recordings
└── logs/
    ├── base/             # Input .rrd recordings
    └── estimate/         # Estimated trajectories and summaries
```

Run all commands from the repository root.

## Requirements

- Python 3.10+
- NumPy
- SciPy
- OpenCV
- PyYAML
- Rerun SDK
- pupil-apriltags
- pandas

The required Python packages are listed in `config/requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r config/requirements.txt
```

## Input data

The main input is a Rerun `.rrd` recording containing:

- `/camera/rear` — JPEG-encoded or raw rear-camera images
- `/imu/accel/x`, `/imu/accel/y`, `/imu/accel/z`
- `/imu/gyro/x`, `/imu/gyro/y`, `/imu/gyro/z`
- A timeline named `time`

The rear-camera frames are expected to be square fisheye frames. The supplied calibration uses `1920 × 1920`; the loader will scale intrinsics if the actual frame resolution differs, provided the scaling is uniform.

If starting from an Insta360 `.insv` file:

```bash
python converter.py input/60Hz.insv
```

This writes `logs/base/60Hz.rrd`. 


## Quick start

Run the complete VIO pipeline:

```bash
python optimize.py --rrd logs/base/60Hz.rrd
```

`--rrd` is required. Defaults for the calibration, anchor, front-end, and optimizer come from `config/vio.yaml`.

Outputs are named from the input recording stem:

```text
logs/estimate/60Hz_trajectory.npz
logs/estimate/60Hz_vio.rrd
```

Open the trajectory in Rerun:

```bash
rerun logs/estimate/60Hz_vio.rrd
```

The output `.rrd` contains only:

```text
/world/cam1_rear_trajectory
```

## Configuration

`config/vio.yaml` controls the runtime parameters:

```yaml
anchor:
  id: 0
  size_m: 0.100

frontend:
  max_corners_per_cam: 220
  min_track_len: 5

optimizer:
  max_iters: 40
  anchor_sigma_px: 0.05
  klt_sigma_px: 0.40
  anchor_huber_px: 10.0
  klt_huber_px: 2.5
  init_prior_rot: 0.0
  init_prior_trans: 0.0
```

### Optimizer parameters

| Key | Meaning | Default |
|---|---|---:|
| `max_iters` | Maximum Levenberg-Marquardt iterations | `40` |
| `anchor_sigma_px` | Anchor corner reprojection noise sigma | `0.05` |
| `klt_sigma_px` | KLT reprojection noise sigma | `0.40` |
| `anchor_huber_px` | Huber threshold for anchor residuals | `10.0` |
| `klt_huber_px` | Huber threshold for KLT residuals | `2.5` |
| `init_prior_rot` | Optional PnP branch-continuity rotation prior; `0` disables | `0.0` |
| `init_prior_trans` | Optional PnP branch-continuity translation prior; `0` disables | `0.0` |

### Anchor branch selection

The AprilTag pose can have two planar-PnP branches. The optimizer uses the anchor observations to select the branch:

- `anchor_sigma_px: 0.05` — tight anchor constraint, keeps the continuity-selected low-motion branch
- `anchor_sigma_px: 0.30` — looser anchor constraint, allows the high-motion branch

The default configuration uses the validated low-motion branch.

### Command-line overrides

Any value from `config/vio.yaml` can be overridden on the command line:

```bash
python optimize.py \
  --rrd logs/base/60Hz.rrd \
  --anchor-sigma 0.30 \
  --max-iters 50
```

Use a different YAML file:

```bash
python optimize.py \
  --rrd logs/base/60Hz.rrd \
  --config path/to/other.yaml
```

Set custom output paths:

```bash
python optimize.py \
  --rrd logs/base/60Hz.rrd \
  --out-npz logs/estimate/custom.npz \
  --viz logs/estimate/custom.rrd
```

The YAML loader rejects unknown keys, so misspelled parameter names fail early.

## Calibration

`config/cam-imu.yaml` contains:

- `cam0` and `cam1` intrinsics and distortion coefficients
- `timeshift_cam_imu` for each camera
- `T_cam_imu` extrinsics
- IMU update rate, noise densities, random walks, and gravity magnitude

Although both cameras are present in the file, the VIO pipeline uses `cam1` only. The rear-camera timeshift is applied before static IMU initialization and IMU interval slicing.

`config/anchor-tags.yaml` documents the single anchor tag, but runtime anchor values come from `config/vio.yaml` or `--anchor-id` / `--anchor-size`.

## Pipeline details

### 1. Preprocessing

`preprocess.py`:

- Loads image and IMU streams from the Rerun recording
- Decodes rear-camera frames lazily
- Converts accelerometer values from g to m/s² when needed
- Applies `timeshift_cam_imu` for `cam1`
- Estimates initial gyro bias and gravity alignment from the stationary pre-roll
- Slices the 1000 Hz IMU stream into intervals `[t_k, t_{k+1}]` aligned with rear-camera frames

### 2. Visual front-end

`frontend.py`:

- Detects the configured `tag36h11` anchor
- Refines tag corners in a virtual pinhole image
- Solves the tag pose with fisheye PnP
- Tracks KLT features with gyroscope-aided prediction
- Triangulates static 3D landmarks
- Uses continuity to choose between planar-PnP pose branches

### 3. Batch optimization

`optimize.py`:

- Preintegrates IMU measurements between camera frames
- Initializes per-frame rotation, position, velocity, gyro bias, accelerometer bias, 3D landmarks, and gravity tilt
- Minimizes IMU, anchor, and KLT residuals with on-manifold Levenberg-Marquardt
- Applies Huber weighting to visual residuals
- Outputs `T_world_imu` and the rear-camera trajectory `T_world_cam1`

The world frame is aligned with the stationary anchor tag.

## Output format

The `.npz` trajectory contains:

| Array | Shape | Description |
|---|---|---|
| `t_cam1` | `(N,)` | Rear-camera timestamps |
| `T_world_imu` | `(N, 4, 4)` | IMU pose in world frame |
| `T_world_cam1` | `(N, 4, 4)` | Rear-camera pose in world frame |
| `v_world` | `(N, 3)` | World-frame velocity |
| `b_gyro` | `(N, 3)` | Gyro bias |
| `b_accel` | `(N, 3)` | Accelerometer bias |
| `landmarks_world` | `(M, 3)` | Optimized 3D landmarks |
| `gravity_world` | `(3,)` | Estimated gravity vector |

Load it with:

```python
import numpy as np

data = np.load("logs/estimate/60Hz_trajectory.npz")
poses = data["T_world_cam1"]
positions = poses[:, :3, 3]
```

## Front-end only

To inspect front-end measurements without running the optimizer:

```bash
python frontend.py --rrd logs/base/60Hz.rrd
```

This prints frame count, anchor observation count, initial PnP pose count, and triangulated KLT landmark count.

## Troubleshooting

- **Missing `--rrd`**: the main CLI requires `--rrd`; there is no default recording path.
- **Missing camera or IMU entities**: confirm the `.rrd` contains `/camera/rear` and all six `/imu/...` scalar streams.
- **Static initialization fails**: the recording needs enough stationary IMU samples before the first rear-camera frame.
- **IMU coverage error**: the IMU stream must cover the full shifted rear-camera time range.
- **Unexpected vertical trajectory branch**: try `--anchor-sigma 0.30` and compare the result.
- **Unknown YAML key**: remove or correct the key; the loader rejects unknown parameters.
