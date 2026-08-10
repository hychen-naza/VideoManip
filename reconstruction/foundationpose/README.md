# FoundationPose — our multi-scale object pose estimation

We do **not** vendor the [FoundationPose](https://github.com/NVlabs/FoundationPose)
codebase in this repo. Instead we keep only:

1. **Our own scripts** (`scripts/`) — the multi-scale pose-estimation entry point
   and its Docker runners / visualizers.
2. **A patch** (`foundationpose_modifications.patch`) — the small set of edits we
   made to upstream FoundationPose files (data reader + GPU-memory + build/env).

To use it you clone upstream FoundationPose, apply our patch, drop our scripts in,
and run. Everything below assumes an NVIDIA GPU with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed (we run FoundationPose inside Docker).

---

## What our changes are (and why)

Applied by `foundationpose_modifications.patch`:

| File | Change | Why it's needed |
|------|--------|-----------------|
| `datareader.py` | `YcbineoatReader` gains an `object_type` arg; reads masks from `masks_pred_obj/<frame>_<type>_object.png` and mesh `textured_simple_<type>.obj` | **Required** by `run_demo_multi_scale.py` to handle the `grasp` vs `target` object of a manipulation clip |
| `learning/training/predict_score.py` | inference render batch `512 → 32` | Fit refiner/scorer on a single consumer GPU |
| `learning/training/predict_pose_refine.py` | render batch `512 → 128`, aggressive `empty_cache()` / tensor frees | Same — avoids CUDA OOM during pose refinement |
| `learning/datasets/h5_dataset.py` | `empty_cache()` + CPU-offload of one warp | Same |
| `bundlesdf/mycuda/setup.py` | `c++14 → c++17` | Build the custom CUDA op with newer CUDA |
| `docker/dockerfile` | `cudagl 11.3 → 12.1`, torch `2.0/cu118 → 2.1/cu121` | Match our GPU/driver |
| `docker/run_container.sh` | drop `xhost +` | Headless machines |
| `Utils.py`, `estimater.py`, `run_demo.py`, `training_config.py` | whitespace / `import pdb` / a customized single-object `run_demo.py` (superseded by our multi-scale script) / training batch size | Non-essential; kept for a faithful record |

Only the first six rows matter for running our pipeline; the rest are harmless.

---

## Setup

```bash
# 1. Clone the exact upstream commit we based on
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
git checkout e3d597b8c6b851d053094ebd6fa240191c5238f8

# 2. Apply our modifications
git apply /path/to/reconstruction/foundationpose/foundationpose_modifications.patch
#   (use `git apply --reject` if upstream has drifted, then fix any .rej files)

# 3. Drop our scripts in at the repo root
cp /path/to/reconstruction/foundationpose/scripts/*.py .
cp /path/to/reconstruction/foundationpose/scripts/*.bash .
cp /path/to/reconstruction/foundationpose/scripts/*.sh .
cp /path/to/reconstruction/foundationpose/scripts/docker/run_command_in_docker.sh docker/

# 4. Download FoundationPose model weights (see upstream README) into ./weights
#    https://github.com/NVlabs/FoundationPose#model-weights
#    You need the score + refiner checkpoints; place them where upstream expects.

# 5. Build the Docker image (uses our patched docker/dockerfile)
cd docker && docker build -t foundationpose:latest -f dockerfile .. && cd ..
```

---

## Expected `demo_data` layout

`run_demo_multi_scale.py` reads, per object clip, from
`FoundationPose/demo_data/<object_folder>/`:

```
demo_data/<object_folder>/
  rgb/<frame>.png                                   # RGB frames
  depth/<frame>.png                                 # depth (from moge_depth.py)
  masks_pred_obj/<frame>_<object_type>_object.png   # object masks (from sam2)
  cam_K.txt                                         # 3x3 intrinsics
  mesh_original/textured_simple_<object_type>.obj   # initial mesh (from img2mesh.py)
```

`<object_type>` is either `grasp` (the object being grasped) or `target`.
These inputs are produced by the earlier stages of `process_videos.sh`
(`video_parser.py`, `moge_depth.py`, sam2 masks, `img2mesh.py`).

Outputs are written to:
```
FoundationPose/debug_multi_scale/<object_folder>/<object_type>/   # per-scale renders + multi_scale_results.txt
FoundationPose/demo_data/<object_folder>/mesh/textured_simple_<object_type>.obj   # best-scale mesh
FoundationPose/demo_data/<object_folder>/obj_mesh/               # per-frame posed meshes + object_mesh_poses_<type>.npy
```

---

## Running

`run_demo_multi_scale.py` tries several mesh scales
(`--scale_range`, default `0.6, 0.75, 0.9, 1.0, 1.1`) and keeps the best fit.

```bash
# Directly (inside the container, from the FoundationPose dir):
python run_demo_multi_scale.py --object_folder <name> --object_type grasp
python run_demo_multi_scale.py --object_folder <name> --object_type target
```

Or via our Docker wrappers (run these from the host):

```bash
# Spin up the container, run object_pos_est.bash for the given objects, tear down:
DATA_ROOT=/path/to/reconstruction/data CUDA_DEVICE=0 \
  ./run_object_pos_est_in_docker.sh <object_folder> [<object_folder> ...]

# object_pos_est.bash runs BOTH grasp and target passes for each object.

# Generic "run any command in the container" helper:
./docker/run_command_in_docker.sh python run_demo_multi_scale.py --object_folder <name> --object_type grasp
```

`DATA_ROOT` and `CUDA_DEVICE` are read from the environment (defaulting to
`<repo>/reconstruction/data` and GPU `0`), forwarded into the container, and the
data root is bind-mounted. Nothing needs editing for a new machine. The usual way
to invoke this is through the pipeline driver, which sets both for you:

```bash
cd reconstruction
FOUNDATIONPOSE_DIR=/path/to/FoundationPose ./process_videos.sh --stages obj_pose <object>
```

### Key arguments (`run_demo_multi_scale.py`)
- `--object_folder` — folder name under the data root (default `cloth`)
- `--object_type`   — `grasp` or `target` (default `grasp`)
- `--scale_range`   — comma-separated scale factors to search (default `0.6, 0.75, 0.9, 1.0, 1.1`)
- `--est_refine_iter` (5), `--track_refine_iter` (2), `--debug_dir`
- `--mesh_file`, `--test_scene_dir` — default to `$DATA_ROOT/<object_folder>/...`,
  falling back to FoundationPose's own `demo_data/` when `DATA_ROOT` is unset

If the grasp mesh is missing the script exits non-zero telling you to run stage 6
first; a missing *target* mesh is treated as "this clip has no target object" and
that pass is skipped.

---

## Visualization helpers
- `2d_visualize_object_and_hand_meshes.py` / `3d_visualize_object_and_hand_meshes.py`
  overlay the estimated object mesh (and hand mesh) back onto the frames.
