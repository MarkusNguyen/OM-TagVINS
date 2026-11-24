# OM-TagVINS

Rear-camera-only visual-inertial odometry with a single stationary AprilTag anchor.

The pipeline estimates a metric 6-DoF camera trajectory by combining:

- 1000 Hz IMU preintegration
- AprilTag corner observations of one stationary `tag36h11` anchor
- Gyro-aided KLT feature tracking
- A full-batch GTSAM factor-graph optimizer

Only the rear camera (`cam1`) is used for VIO. The front camera is not used.

## Project layout

```text
.
├── converter.py          # Optional: convert Insta360 .insv to a Rerun .rrd
├── preprocess.py         # Load and synchronize camera/IMU data
├── trajectory.py         # CLI entry point: runs VIO and writes the TCP trajectory
├── frontend.py           # Front-end entry point: wires together the src/ helpers
├── optimize.py           # VIO entry point: wires together the src/ helpers
├── src/                  # Focused implementation modules (see "Source modules" below)
│   ├── constants.py
│   ├── lie_algebra.py
│   ├── camera_model.py
│   ├── gyro.py
│   ├── apriltag.py
│   ├── masking.py
│   ├── ransac.py
│   ├── klt_tracker.py
│   ├── triangulation.py
│   ├── imu_preintegration.py
│   ├── vio_initialization.py
│   ├── vio_solver.py
│   └── rerun_logger.py
├── config/
│   ├── settings.py       # YAML configuration loader
│   ├── cam-imu.yaml      # Camera intrinsics, extrinsics, timeshifts, IMU noise
│   ├── cam-in-tcp.yaml   # Camera-to-TCP extrinsics for trajectory.py
│   ├── vio.yaml          # Runtime VIO parameters
│   └── anchor-tags.yaml  # Single-tag anchor reference
├── input/                # Source recordings
└── logs/
    ├── base/             # Input .rrd recordings
    └── estimate/         # Estimated trajectories and summaries
```

Run all commands from the repository root.

## Workflow: what happens during a run

This VIO pipeline is **AprilTag + KLT**, not ORB-SLAM/ORB feature tracking. It does not use ORB features, ORB descriptors, or descriptor matching. The visual inputs are stationary AprilTag corners and gyro-aided KLT point tracks.

```text
.rrd recording
   ↓
preprocess.py        load camera / IMU / calibration / rolling-shutter metadata
   ↓
frontend.py          AprilTag PnP + KLT tracks + triangulation
   ↓
vio_initialization   full-rate PnP/IMU initialization
   ↓
vio_solver.py        full-batch GTSAM optimization
   ↓
interpolation        restore full camera frame rate
   ↓
trajectory.py        camera → TCP transform + save outputs
```

### 1. Load and prepare data

`preprocess.py` reads the rear camera and IMU streams from the `.rrd`.

- Camera frames are streamed as encoded JPEG blobs and decoded lazily.
- IMU samples are sliced into one interval between each pair of camera frames.
- Camera intrinsics, camera-IMU extrinsics, gyro/accel corrections, and noise parameters come from `config/cam-imu.yaml`.
- Rolling-shutter readout comes from `.insv` metadata through `converter.py`.

For the supplied base recording, the readout is:

```text
0.015727346420288085 s
```

Rolling-shutter compensation stays enabled. Camera-minus-IMU time-offset estimation is currently disabled:

```yaml
rolling_shutter_readout_s: null   # null = use .insv metadata
estimate_time_offset: false
```

### 2. Front-end frame loop

`frontend.py` processes frames in timestamp order.

For each frame:

1. Detect the AprilTag anchor.
2. Refine tag corners using the full-resolution image.
3. Run rolling-shutter-compensated fisheye PnP.
4. Use the previous tag pose to reject the wrong planar-PnP branch.
5. Predict KLT feature locations using gyro relative rotation.
6. Track corners from the previous frame into the current frame with optical flow.
7. Reject bad tracks with forward/backward flow and epipolar checks.
8. Retire dead tracks and replenish new corners.
9. Triangulate static KLT tracks into 3D landmarks.

The tracker uses previous-frame values in memory:

```text
previous image → previous corner positions → gyro prediction → current frame
```

These previous values are not saved as keyframes. They exist only while the frontend is running.

### 3. Initialization

`src/backend/vio_initialization.py` initializes the batch state at full frame rate:

- poses from AprilTag PnP
- velocities from PnP positions and IMU preintegration
- Gyro and accelerometer biases
- 3D landmark positions
- gravity direction
- anchor world pose

The velocity refinement uses a sparse least-squares solve, which avoids the old multi-GB dense matrix on long recordings.

### 4. Backend optimization

`src/backend/vio_solver.py` builds one GTSAM batch graph containing:

- combined IMU factors
- anchor reprojection factors
- KLT reprojection factors
- bias priors and bias random-walk factors
- an optimizable anchor pose with a tilt-free gauge prior
- rolling-shutter-aware custom visual factors

The anchor is not fixed at identity. Its position and yaw are pinned by a gauge prior, while tilt is free because gravity makes it observable.

Time-offset estimation is disabled for the current test:

```yaml
estimate_time_offset: false
```

### 5. Frame subsampling, not keyframes

There is no keyframe system in this repo.

The backend uses a fixed stride:

```yaml
backend_frame_stride: 2
```

That means GTSAM optimizes every second frame:

```text
6933 camera frames → ~3467 optimized backend frames
```

This is **not** adaptive keyframe selection. It does not choose frames by parallax, motion, anchor visibility, or information gain. It is just uniform subsampling to reduce the GTSAM state size.

After optimization, the result is interpolated back to the original full camera frame rate.

Keep `backend_frame_stride` at `2`. Larger values merge IMU intervals into very long segments and LM can fail immediately.

### 6. Does a revisited frame reuse previous values?

There is no loop closure or frame revisiting logic.

- **KLT:** yes, each new frame uses the previous frame’s tracked corners and rays.
- **AprilTag PnP:** yes, the previous tag pose is used for branch continuity.
- **GTSAM backend:** no, it is one full batch optimization. It does not incrementally reuse a previously optimized keyframe value.
- **Saved keyframes:** no.

`init_prior_rot` and `init_prior_trans` are currently `0.0`, so previous PnP poses are not added as strong pose priors in the optimizer.

### 7. Outputs

The VIO stage saves a full-rate trajectory:

```text
t_cam1
T_world_cam1
T_world_imu
v_world
b_gyro
b_accel
landmarks_world
T_world_tag
```

Then `trajectory.py` converts camera poses to TCP poses and writes:

```text
<recording>_tcp_trajectory.npz
<recording>_tcp_vio.rrd
```

## Source modules (`src/`)

The implementation is split into small, focused modules under `src/`. The top-level
`frontend.py` and `optimize.py` are import-only entry points: they contain no CLI
parser and are not run directly. They import these modules and expose
`run_frontend()` and `run_vio()` respectively. Modules in `src/` never import
`frontend.py` or `optimize.py`, so the dependency direction is one-way and each
module can be edited independently.

The modules are grouped by responsibility:

| Module | Responsibility | Key functions / classes |
|---|---|---|
| `src/core/constants.py` | Shared rear-camera identifiers used by the front-end modules. | `REAR_CAM`, `ACTIVE_CAMS`, `CAMERA_IMAGE_KEYS` |
| `src/core/lie_algebra.py` | SO(3) math: rotation logarithm/exponential and right Jacobians used by the IMU preintegrator and solver. | `skew`, `so3_exp`, `so3_log`, `so3_right_jacobian`, `so3_inv_right_jacobian` |
| `src/core/camera_model.py` | Kannala-Brandt fisheye model: forward projection, inverse unprojection to unit bearing rays, and the analytical projection Jacobian. | `project_kannala_brandt`, `unproject_kannala_brandt`, `project_kb_with_jacobian` |
| `src/frontend/gyro.py` | Integrates 1000 Hz gyro samples, converts IMU relative rotation into camera-frame rotation, and maps rolling-shutter rays back to the nominal frame timestamp. | `integrate_gyro_rotation`, `compute_camera_relative_rotation`, `compensate_rolling_shutter_rays` |
| `src/frontend/apriltag.py` | Anchor tag geometry, IPPE fisheye pose solving with branch continuity, and sub-pixel corner refinement in a virtual pinhole image. | `get_tag_corners_3d`, `solve_tag_pose_fisheye`, `refine_tag_corners_virtual_pinhole` |
| `src/frontend/masking.py` | Builds the fisheye aperture mask and blocks static foreground regions (gripper / operator body). | `build_static_masks` |
| `src/frontend/ransac.py` | Gyro-aided 2-point epipolar test that rejects dynamic outliers while retaining low-parallax static background points. | `ransac_2pt_gyro_epipolar` |
| `src/frontend/klt_tracker.py` | IMU-aided fisheye KLT tracker: gyro-predicted initial flow, forward/backward check, outlier rejection, and new-feature replenishment. | `IMUAidedFisheyeKLTTracker` |
| `src/frontend/triangulation.py` | Linear DLT triangulation directly on bearing rays, plus reprojection/parallax filtering to seed initial 3D landmarks. | `triangulate_bearing_rays`, `triangulate_klt_tracks` |
| `src/backend/imu_preintegration.py` | On-manifold IMU preintegration with 9×9 covariance, gyro/accel corrections, and first-order bias Jacobians. | `PreintegratedIMUFactor` |
| `src/backend/vio_initialization.py` | Seeds per-frame poses/velocities/biases, landmark positions, and gravity orientation from PnP + IMU data. | `initialize_vio_states`, `refine_initial_velocities`, `extract_frame_gyros`, `pack_visual_observations` |
| `src/backend/vio_solver.py` | GTSAM batch factor-graph optimizer: IMU, anchor, KLT, bias-prior, and bias-random-walk factors with rolling-shutter-aware custom visual factors, Huber robust noise, an optimizable anchor pose with a tilt-free gauge prior, and gravity-aligned output. | `huber_irls_weights`, `evaluate_visual_residuals_and_jacobians`, `run_batch_vio_optimization` |
| `src/viz/rerun_logger.py` | Writes the optimized rear-camera trajectory and per-frame pose axes to a Rerun `.rrd`. | `log_optimized_vio_to_rerun` |

If you want to change how a specific part of the pipeline behaves, edit the
corresponding module rather than the entry points — for example, tag detection
in `src/frontend/apriltag.py`, feature tracking in `src/frontend/klt_tracker.py`,
or factor weights and optimizer convergence in `src/backend/vio_solver.py`.

## Requirements

- Python 3.12 recommended
- NumPy
- SciPy
- OpenCV
- PyYAML
- Rerun SDK with the `catalog` extra
- pupil-apriltags
- pandas
- telemetry-parser
- GTSAM

Dependencies are declared in `pyproject.toml`. Install Python 3.12 and create the environment with `uv`:

```powershell
uv python install 3.12
uv sync
```

Then run the pipeline with:

```powershell
uv run python -u trajectory.py --rrd "logs\base\VID_20261117_145759_00_025.rrd"
```

`uv sync` creates `.venv/` and `uv.lock` from `pyproject.toml`. `config/requirements.txt` is kept only as an optional pip-compatible reference.

## Input data

The main input is a Rerun `.rrd` recording containing:

- `/camera/rear` — JPEG-encoded or raw rear-camera images
- `/imu/accel/x`, `/imu/accel/y`, `/imu/accel/z`
- `/imu/gyro/x`, `/imu/gyro/y`, `/imu/gyro/z`
- A timeline named `time`

The rear-camera frames are expected to be square fisheye frames. The supplied calibration uses `1920 × 1920`; the loader will scale intrinsics if the actual frame resolution differs, provided the scaling is uniform.

If starting from an Insta360 `.insv` file:

```bash
uv run python converter.py input/60Hz.insv
```

This writes `logs/base/60Hz.rrd`. Conversion uses `telemetry-parser` to read
`Lens.Data.frame_readout_time` from the `.insv` metadata (the same field used by
specialist-ai/specialist), converts milliseconds to seconds, and stores it on
the recording as `/calibration/rolling_shutter_readout_s`. The VIO loader reads
that entity automatically; an explicit value in `config/vio.yaml` or on the CLI
overrides it.


## Quick start

The command-line entry point is `trajectory.py`. It runs VIO through
`optimize.run_vio()` and converts the resulting camera trajectory to a TCP
trajectory:

```bash
uv run python trajectory.py --rrd logs/base/60Hz.rrd
```

With `--rrd`, VIO runs first. With `--npz`, an existing camera trajectory is
read instead and only the TCP transform is applied. With neither flag, the
newest `*_trajectory.npz` in `logs/estimate/` is used. Defaults for the
calibration, anchor, front-end, and optimizer come from `config/vio.yaml`.

Outputs are named from the input recording stem:

```text
logs/estimate/60Hz_tcp_trajectory.npz
logs/estimate/60Hz_tcp_vio.rrd
```

The `.npz` contains `t_tcp`, `T_world_tcp`, `T_tcp0_tcp`, `T_world_cam1`,
`T_cam_tcp`, and `T_tcp_cam`. The intermediate VIO result also stores
`td_cam_imu` (estimated camera-minus-IMU offset, seconds),
`rolling_shutter_readout_s`, and the optimized reference anchor pose
`T_world_tag`. Open the Rerun recording with:

```bash
uv run rerun logs/estimate/60Hz_tcp_vio.rrd
```

The TCP `.rrd` contains:

```text
world/tcp_trajectory    # static line strip
world/tcp_pose/axes     # per-timestamp pose axes, logged on the "time" timeline
world/tcp_marker        # moving marker, logged on the "time" timeline
```

The VIO `.rrd` written by `optimize.py` additionally contains
`world/cam1_pose/axes` and `world/cam1_pose/origin` at each camera timestamp.

Use `--no-viz` to skip writing the `.rrd`.

The VIO stage can also be used directly as a library (there is no CLI parser in
`optimize.py` or `frontend.py`):

```python
from optimize import run_vio

result = run_vio("logs/base/60Hz.rrd")   # returns T_world_cam1, T_world_imu, ...
```

## Configuration

`config/vio.yaml` controls the runtime parameters:

```yaml
anchor:
  id: 0
  size_m: 0.100

frontend:
  # Specialist-matched frontend values.
  # These are much heavier than the previous runtime-tuned values.
  max_corners_per_cam: 4000
  min_track_len: 3
  min_parallax_deg: 0.5
  max_track_len: 300
  max_triangulation_obs: 8
  tag_detect_stride: 1
  tag_detect_scale: 0.5

optimizer:
  # Specialist SolverCfg.max_iterations.
  max_iters: 30
  # Batch-graph size controls (initialization still uses all detections)
  anchor_factor_stride: 4
  max_klt_obs_per_landmark: 30
  backend_frame_stride: 2
  anchor_sigma_px: 1.05
  klt_sigma_px: 0.5
  anchor_huber_px: 1.41
  klt_huber_px: 0.67
  klt_omega_rate_scale: 0.0
  rolling_shutter_readout_s: null
  estimate_time_offset: false
  time_offset_prior_s: 0.005
  init_prior_rot: 0.0
  init_prior_trans: 0.0
```

### Frontend parameters

| Key | Meaning | Default |
|---|---|---:|
| `max_corners_per_cam` | Maximum tracked KLT corners; Specialist `max_corners` | `4000` |
| `min_track_len` | Minimum KLT track length in frames; Specialist `min_track_length` | `3` |
| `min_parallax_deg` | Minimum KLT bearing parallax for triangulation | `0.5` |
| `max_track_len` | Maximum observations retained per KLT track; numeric match to Specialist `max_track_span` (keyframes) | `300` |
| `max_triangulation_obs` | Observations sampled per track for DLT triangulation; numeric match to Specialist `max_obs_per_track` | `8` |
| `tag_detect_stride` | Run AprilTag detection every N-th frame (keep at `1` for stable PnP initialization) | `1` |
| `tag_detect_scale` | Detection image scale; corners are still refined at full resolution | `0.5` |

For long recordings, the front end stores compact numeric observations instead of one NumPy array per point per frame, retires dead KLT tracks, and caps DLT systems. The five frontend/solver values above now match the corresponding Specialist defaults. They are heavier than the previous runtime-tuned values (`220 / 5 / 90 / 24 / 8`), so expect longer frontend and optimizer times. Full-resolution AprilTag detection is the largest per-frame cost, so detection runs on a half-resolution image and refines the returned corners against the full-resolution frame. Increasing `tag_detect_stride` further speeds up detection but makes the anchor-based initialization too sparse and can destabilize the batch optimization.

### Optimizer parameters

| Key | Meaning | Default |
|---|---|---:|
| `max_iters` | Maximum optimizer iterations; Specialist `max_iterations` | `30` |
| `anchor_factor_stride` | Add anchor gauge factors only every N-th optimized frame | `4` |
| `max_klt_obs_per_landmark` | Maximum KLT observations per landmark in the batch graph | `30` |
| `backend_frame_stride` | Optimize every N-th camera frame and interpolate the full-rate output | `2` |
| `anchor_sigma_px` | Anchor corner reprojection noise sigma | `1.05` |
| `klt_sigma_px` | KLT reprojection noise sigma | `0.5` |
| `anchor_huber_px` | Huber threshold for anchor residuals | `1.41` |
| `klt_huber_px` | Huber threshold for KLT residuals | `0.67` |
| `klt_omega_rate_scale` | Downweight fast-rotation KLT observations by `1 + scale*||omega||` | `0.0` |
| `rolling_shutter_readout_s` | Full-frame readout in seconds; `null` uses `.insv` metadata, `0` disables | `null` |
| `estimate_time_offset` | Estimate camera-minus-IMU time offset as an optimizer variable | `false` |
| `time_offset_prior_s` | Prior sigma (seconds) on that time offset | `0.005` |
| `init_prior_rot` | Optional PnP branch-continuity rotation prior; `0` disables | `0.0` |
| `init_prior_trans` | Optional PnP branch-continuity translation prior; `0` disables | `0.0` |

The full batch graph uses one Python `CustomFactor` per visual observation, so factor count dominates LM runtime. Anchor factors are subsampled with `anchor_factor_stride` and KLT observations are sampled evenly up to `max_klt_obs_per_landmark`; initialization still uses every anchor sighting. `backend_frame_stride` reduces the number of optimizer states by keeping every N-th frame. State initialization is still done at full frame rate before downsampling, and the returned trajectory is interpolated back to the original frame rate. Keep this value at `2`: larger values merge IMU intervals into very long segments, and LM can fail to find a descent direction on the resulting graph.

### Anchor branch selection

The AprilTag pose can have two planar-PnP branches. The frontend keeps only branches whose reprojection error is within 3 px of the best branch, then applies continuity. Anchor corner noise and weighting affect how strongly the optimizer follows the initialized branch versus smoothing observations:

- `anchor_sigma_px: 1.05` (default) — stronger tag pull; overfitting can couple tag jitter into translation
- `anchor_sigma_px: 0.5` — looser constraint; smoother but less influenced by the anchor
- `init_prior_rot` / `init_prior_trans` — optional priors to keep poses near the continuity-selected PnP initialization

### Command-line overrides

Any value from `config/vio.yaml` can be overridden on the command line:

```bash
uv run python trajectory.py \
  --rrd logs/base/60Hz.rrd \
  --anchor-sigma 0.30 \
  --min-parallax-deg 0.15 \
  --max-track-len 90 \
  --tag-detect-scale 0.5 \
  --max-iters 50 \
  --rolling-shutter-readout-s 0.004 \
  --time-offset-prior-s 0.005 \
  --no-estimate-time-offset
```

Use a different YAML file:

```bash
uv run python trajectory.py \
  --rrd logs/base/60Hz.rrd \
  --vio-config path/to/other.yaml
```

Set custom output paths:

```bash
uv run python trajectory.py \
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
- IMU update rate, gyro sign, noise densities, random walks, and gravity magnitude

Although both cameras are present in the file, the VIO pipeline uses `cam1` only. The rear-camera timeshift is applied before static IMU initialization and IMU interval slicing.

`config/anchor-tags.yaml` documents the single anchor tag, but runtime anchor values come from `config/vio.yaml` or `--anchor-id` / `--anchor-size`.

## Pipeline details

The entry points below delegate the actual work to the modules listed in
[Source modules](#source-modules-src).

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
- Minimizes IMU, anchor, and KLT residuals with GTSAM Levenberg-Marquardt
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
| `gravity_world` | `(3,)` | Estimated gravity vector, approximately `[0, 0, -9.81]` in the returned frame |
| `R_gravity_align` | `(3, 3)` | Rotation applied to map the internal tag frame to the returned gravity-aligned frame |

Load it with:

```python
import numpy as np

data = np.load("logs/estimate/60Hz_trajectory.npz")
poses = data["T_world_cam1"]
positions = poses[:, :3, 3]
```

## Front-end only

`frontend.py` is an import-only module, so call `run_frontend()` to inspect
front-end measurements without running the optimizer:

```python
from frontend import run_frontend

visual_meas = run_frontend("logs/base/60Hz.rrd")
```

This prints the frame count, anchor observation count, initial PnP pose count,
and triangulated KLT landmark count, and returns the `anchor_obs`,
`init_poses_world_imu`, `klt_tracks`, and `landmarks_world` arrays.

## Troubleshooting

- **No input specified**: `trajectory.py` needs `--rrd`, `--npz`, or an existing `*_trajectory.npz` in `logs/estimate/`. It does not use a default recording path.
- **Missing camera or IMU entities**: confirm the `.rrd` contains `/camera/rear` and all six `/imu/...` scalar streams.
- **Static initialization fails**: the recording needs enough stationary IMU samples before the first rear-camera frame.
- **IMU coverage error**: the IMU stream must cover the full shifted rear-camera time range.
- **Unexpected vertical trajectory branch**: try `--anchor-sigma 0.30` and compare the result.
- **Unknown YAML key**: remove or correct the key; the loader rejects unknown parameters.
