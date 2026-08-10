# VideoManip — video-to-manipulation reconstruction pipeline

Reconstruct hand + object manipulation from a single RGB video: extract frames,
segment the object, estimate depth & camera intrinsics, recover the hand mesh,
reconstruct the object mesh, estimate 6-DoF object pose over time, and retarget
the human hand onto a robot hand.

This repo contains **our pipeline scripts** plus **thin wrappers** around several
open-source repos. The external repos are **pinned git submodules** nested inside
wrapper folders, each with a patch of our edits and the extra files we wrote.
The reconstruction pipeline lives in `reconstruction/`; the optional ContactOpt
grasp-refinement component lives in `grasping/`. See
[`EXTERNAL_REPOS.md`](EXTERNAL_REPOS.md) for the submodule details.

A small **example video** ships with the repo, so you can run the pipeline
end-to-end immediately after setup — no data collection required.

---

## 1. Quick start

```bash
git clone --recursive https://github.com/hychen-naza/VideoManip.git
cd VideoManip

conda env create -f reconstruction/video_env.yml   # creates the `video` env
conda activate video

./reconstruction/setup_externals.sh                # patches submodules, builds hand meshes
# download the model weights listed in section 3, then:

cd reconstruction
./process_videos.sh                                # runs the bundled example video
```

That processes `reconstruction/example_data/real_14_pourtea.mp4` and writes
everything to `reconstruction/data/real_14_pourtea/`.

**Nothing is hardcoded to a particular machine.** Every path is resolved
relative to the repository, and anything machine-specific is a flag or an
environment variable.

---

## 2. Repository layout

```
reconstruction/                    the reconstruction pipeline (this folder)
  process_videos.sh                the pipeline driver -- run this
  example_data/                    bundled example input video
  data/                            per-object outputs (created at run time, gitignored)
  utils/                           helper scripts run by the driver:
    video_parser.py                stage 1  video -> RGB frames
    moge_depth.py                  stage 3  depth + camera intrinsics
    img2mesh.py                    stage 6  object mesh from cropped frames
    mesh_traj_render.py, pcd_vis_utils.py    rendering / point-cloud helpers
  sam2/                stages 2 & 5  clicks + object masks             (SAM 2)
  hamer/               stage 4  hand mesh                              (HaMeR)
  foundationpose/      stage 7  6-DoF object pose (Docker)             (FoundationPose)
  dex_retargeting/     stage 8  hand -> robot retargeting              (dex-retargeting)
    prepare_hand_meshes.py         generates robot visual meshes from the assets submodule
  setup_externals.sh               inits submodules, applies patches, copies our files
  EXTERNAL_REPOS.md                how the submodules + our patches are organized
grasping/
  contactopt/          optional grasp/contact refinement               (ContactOpt)
```

Each wrapper folder has its own README explaining how to install that component.

---

## 3. Setup

### 3.1 Clone with submodules
```bash
git clone --recursive https://github.com/hychen-naza/VideoManip.git
cd VideoManip
# if you forgot --recursive:
git submodule update --init --recursive
```

### 3.2 Python environment
```bash
conda env create -f reconstruction/video_env.yml   # creates the `video` env
conda activate video
```
Some components (FoundationPose, ContactOpt) need their own environment — see
each wrapper's `README.md`.

### 3.3 Apply our code onto the submodules
```bash
./reconstruction/setup_externals.sh
```
This initializes each submodule, applies our patch, copies our added files in,
and generates the robot-hand visual meshes. It is **idempotent** — re-running it
detects already-applied patches instead of erroring — and it reports every
component that failed rather than stopping at the first one. Run it from
anywhere; it locates the repo root itself.

### 3.4 Download weights and licensed assets
These are **not** shipped here:

| What | Where it goes | Source |
|------|---------------|--------|
| MANO models (`mano_v1_2`) | `reconstruction/hamer/hamer/_DATA/` | https://mano.is.tue.mpg.de |
| HaMeR + ViTPose checkpoints | `reconstruction/hamer/hamer/_DATA/` | `reconstruction/hamer/README.md` |
| SAM 2 checkpoint | `reconstruction/sam2/sam2/checkpoints/` | `cd` there and run `./download_ckpts.sh` |
| FoundationPose weights + Docker image | external checkout | `reconstruction/foundationpose/README.md` |
| DRO-Grasp (robot URDFs / point clouds) | sibling checkout, or `DRO_GRASP_ROOT` | needed by stage 8 only |

MoGe (stage 3) downloads its own weights from Hugging Face on first run.

---

## 4. Running the pipeline

```bash
cd reconstruction
./process_videos.sh [options] [object ...]
```

With no arguments it processes **every** video in `$VIDEO_DIR` (the bundled
example by default). Object names are the input filenames without the extension.

### Options

| Flag | Meaning | Default |
|------|---------|---------|
| `--data-root DIR` | Per-object working dirs | `reconstruction/data` |
| `--video-dir DIR` | Input videos / images | `reconstruction/example_data` |
| `--cuda-device N` | GPU index | `0` |
| `--frame-rate N` | Keep every Nth frame | `1` |
| `--rotate` | Rotate frames 90° clockwise on extraction | off |
| `--stages LIST` | Comma-separated stages to run | all but `obj_pose` |
| `--list-stages` | Print stage names and exit | — |
| `-h, --help` | Usage | — |

Every flag also works as an environment variable (`DATA_ROOT`, `VIDEO_DIR`,
`CUDA_DEVICE`, `FRAME_RATE`, `ROTATE_FRAMES`), plus `FOUNDATIONPOSE_DIR`,
`DRO_GRASP_ROOT`, `MESHY_API_KEY`, `OPENAI_API_KEY`, `USE_OPTIMIZE`.

### Examples
```bash
./process_videos.sh                                   # bundled example
./process_videos.sh --stages frames,intrinsics        # just the first stages
./process_videos.sh --cuda-device 1 real_14_pourtea   # one object on GPU 1
VIDEO_DIR=~/my_videos ./process_videos.sh my_object   # your own input
```

### Input format
One video (or image) per object, named after the object:
```
example_data/
  real_14_pourtea.mp4     # or .avi/.mov/.mkv/.png/.jpg
```
The filename stem (`real_14_pourtea`) is the object key used throughout.

---

## 5. The stages

Stages are **ordered by dependency**; the driver runs them in sequence. Each
writes into `$DATA_ROOT/<object>/`.

| # | Stage | Script | Produces | Needs |
|---|-------|--------|----------|-------|
| 1 | `frames` | `utils/video_parser.py` | `rgb/` | input video |
| 2 | `clicks` | `sam2/sam2/get_click_info.py` | click prompts | `rgb/` |
| 3 | `intrinsics` | `utils/moge_depth.py` | `cam_info/`, `depth/`, `cam_K.txt` | `rgb/` |
| 4 | `hand_mesh` | `hamer/hamer/hand_detection.py` | `human_hand/` | `rgb/`, `cam_K.txt` |
| 5 | `masks` | `sam2/sam2/run_video_function.py` | `masks_pred_obj/`, `croped_frames/` | `rgb/`, clicks |
| 6 | `obj_mesh` | `utils/img2mesh.py` | `mesh_original/` | `croped_frames/`, API keys |
| 7 | `obj_pose` | `foundationpose/` (Docker) | `obj_mesh/` posed meshes | mask, depth, K, mesh |
| 8 | `retarget` | `dex_retargeting/.../retarget_hand_object_common_inspire.py` | `robot_qpos/` | `human_hand/`, DRO-Grasp |

Three stages have prerequisites the pipeline cannot satisfy on its own. Rather
than failing, they **skip with instructions** so the rest of the run completes:

- **Stage 2 (`clicks`)** is interactive — it opens a matplotlib window where you
  left-click the object, right-click background, middle-click to move to the next
  object. Prompts are cached in `sam2/sam2/object_click.json` and reused, so this
  is a no-op for the example video and for any object you have already clicked.
  On a headless machine with no cached prompts it skips and tells you the command
  to run on a GUI machine.
- **Stage 6 (`obj_mesh`)** calls the **Meshy** image-to-3D API and the **OpenAI**
  API, both paid. Export `MESHY_API_KEY` and `OPENAI_API_KEY` to run it, or drop
  your own mesh into `$DATA_ROOT/<object>/mesh/` and skip it.
- **Stage 7 (`obj_pose`)** runs in Docker against an external FoundationPose
  checkout. It is **not** in the default stage list; enable it with
  `FOUNDATIONPOSE_DIR=/path/to/FoundationPose ./process_videos.sh --stages obj_pose`.

---

## 6. Step-by-step walkthrough

Run the whole thing at once with `./process_videos.sh`, or walk it one stage at a
time as below. Each step lists the exact command, what it writes, and how to
check it worked. Commands assume `conda activate video` and `cd reconstruction`.
`OBJ` is the object name — `real_14_pourtea` for the bundled example.

```bash
export OBJ=real_14_pourtea
```

### Step 1 — Extract frames
```bash
./process_videos.sh --stages frames $OBJ
```
Writes `data/$OBJ/rgb/frame_%06d.png` plus `extraction_summary.txt`.
Check: `ls data/$OBJ/rgb/*.png | wc -l` → **148** for the example.
Use `--frame-rate N` to keep every Nth frame, `--rotate` for portrait footage.

### Step 2 — Click the objects (interactive, one-time per object)
```bash
./process_videos.sh --stages clicks $OBJ
```
The example already has saved prompts, so this is a no-op and prints
`reusing saved click prompts`. For **your own** video it opens the first frame:

- **left click** — a point *on* the object
- **right click** — a point on the background
- **middle click** — finish this object, start the next one
- **middle click again** (with no points) — done; press `q` to close

The first object you click is the `grasp_object`, the second the `target_object`.
Prompts are appended to `sam2/sam2/object_click.json` and reused forever after.

> Needs a display. On a headless box the stage skips and prints the command to
> run on a GUI machine; copy the resulting `object_click.json` entry over.

### Step 3 — Depth and camera intrinsics
```bash
./process_videos.sh --stages intrinsics $OBJ
```
Runs MoGe-2 (downloads its weights from Hugging Face on first use).
Writes `data/$OBJ/depth/*.png` (16-bit, millimetres), `data/$OBJ/cam_info/` and
`data/$OBJ/cam_K.txt`.
Check: `cat data/$OBJ/cam_K.txt` → a 3×3 matrix; fx ≈ 808 for the example.

### Step 4 — Hand mesh (HaMeR)
```bash
./process_videos.sh --stages hand_mesh $OBJ
```
Needs `cam_K.txt` from step 3 and the HaMeR/ViTPose/MANO weights in
`hamer/hamer/_DATA/`. Writes `data/$OBJ/human_hand/<frame>_mano_data.npy`
(MANO params, betas, `cam_t`, retarget-ready pose/joints) and `<frame>_all.jpg`
overlays.
Check the last log line: `hand reconstruction finished: 148/148 frames written`.
Frames with no confident right-hand detection are skipped and counted there — a
low ratio means the hand is occluded or out of frame.

### Step 5 — Object masks (SAM 2)
```bash
./process_videos.sh --stages masks $OBJ
```
Propagates the step-2 clicks through the clip. Writes
`data/$OBJ/masks_pred_obj/<frame>_<grasp|target>_object.png` and the cropped
images stage 6 needs in `data/$OBJ/croped_frames/`.
Check: `ls data/$OBJ/masks_pred_obj | wc -l` → **296** (148 frames × 2 objects).

### Step 6 — Object mesh (Meshy + OpenAI, paid)
```bash
export OPENAI_API_KEY=sk-...
export MESHY_API_KEY=msy_...
./process_videos.sh --stages obj_mesh $OBJ
```
Sends the largest-area cropped image to OpenAI to inpaint the occluded object,
then to Meshy's image-to-3D API. Writes
`data/$OBJ/mesh_original/textured_simple_<grasp|target>.obj`.

**Without the keys this stage skips.** To continue without it, drop your own mesh
in yourself — anything trimesh can load works:
```bash
mkdir -p data/$OBJ/mesh_original
cp my_object.obj data/$OBJ/mesh_original/textured_simple_grasp.obj
```

### Step 7 — Object pose over time (FoundationPose, Docker)
Not in the default stage list. One-time setup per machine, per
[`foundationpose/README.md`](foundationpose/README.md): clone upstream
FoundationPose at `e3d597b`, apply `foundationpose_modifications.patch`, copy
`foundationpose/scripts/*` to its root, download the weights, and build the image
(`docker build -t foundationpose:latest -f docker/dockerfile ..`). Then:
```bash
FOUNDATIONPOSE_DIR=/path/to/FoundationPose \
  ./process_videos.sh --stages obj_pose $OBJ
```
Needs steps 3, 5 and 6 (depth, masks, mesh). It searches mesh scales
(`0.6 … 1.1`), keeps the best fit, and writes per-frame posed meshes to
`data/$OBJ/obj_mesh/` plus the chosen mesh to `data/$OBJ/mesh/`.
Check: `ls data/$OBJ/obj_mesh | wc -l` → ~2 files per frame.

### Step 8 — Retarget onto a robot hand
```bash
DRO_GRASP_ROOT=/path/to/DRO-Grasp ./process_videos.sh --stages retarget $OBJ
```
`DRO_GRASP_ROOT` defaults to a `DRO-Grasp` checkout sitting next to this repo.
Needs step 4's `human_hand/`. Writes
`data/$OBJ/robot_qpos/inspire/robot_qpos.npy` and `robot_qpos_index.npy`.
Check:
```bash
python -c "import numpy as np; print(np.load('data/$OBJ/robot_qpos/inspire/robot_qpos.npy').shape)"
# (148, 18)  ->  148 frames x 18 DoF
```
Set `USE_OPTIMIZE=true` for the slower optimization-based variant, which writes
`robot_qpos_optimized.npy` instead.

### Doing it all in one go
```bash
export OPENAI_API_KEY=... MESHY_API_KEY=... 
export FOUNDATIONPOSE_DIR=/path/to/FoundationPose DRO_GRASP_ROOT=/path/to/DRO-Grasp
./process_videos.sh --stages frames,clicks,intrinsics,hand_mesh,masks,obj_mesh,obj_pose,retarget $OBJ
```

---

## 7. Output and logs

```
data/<object>/
  rgb/               extracted frames
  depth/             16-bit depth maps (mm, FoundationPose-compatible)
  cam_info/          per-frame intrinsics;  cam_K.txt holds the first frame's K
  human_hand/        per-frame MANO params + overlay renders (*_mano_data.npy)
  masks_pred_obj/    per-frame object masks
  croped_frames/     cropped object images used for mesh generation
  mesh_original/     generated object mesh              (stage 6)
  obj_mesh/          posed object meshes                (stage 7)
  robot_qpos/<hand>/ retargeted robot trajectory        (stage 8)
  logs/              one log file per stage
```

Every stage is timed and logged to `data/<object>/logs/<NN>_<stage>.log`, and the
driver prints a summary:

```
  STAGE        STATUS     TIME  LOG
  frames       PASS         2s  .../logs/01_frames.log
  clicks       PASS         0s  .../logs/02_clicks.log
  intrinsics   PASS        31s  .../logs/03_intrinsics.log
  hand_mesh    PASS       124s  .../logs/04_hand_mesh.log
  masks        PASS        18s  .../logs/05_masks.log
  obj_mesh     SKIP         0s  .../logs/06_obj_mesh.log
  retarget     PASS        53s  .../logs/08_retarget.log
```

A failing stage does **not** abort the run — it is recorded and the pipeline
continues, so one invocation surfaces every problem. The exit code is the number
of failed stages. `SKIP` means a prerequisite was missing on purpose (see
section 5); the log says exactly what to install or export.

### Reference timings
The bundled example (148 frames, 1280×720, ~5 s) on one RTX 4090:
frames 2 s · intrinsics 31 s · hand_mesh 124 s · masks 18 s · retarget 53 s.

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'mmpose'` / `'dex_retargeting'`**
Those packages are normally `pip install -e`'d, and an editable install records
an absolute path that breaks when the checkout moves. `process_videos.sh` puts
the in-repo copies on `PYTHONPATH`, so run stages through the driver rather than
invoking the Python scripts directly. To run one by hand, export it yourself:
```bash
export PYTHONPATH=reconstruction/hamer/hamer/third-party/ViTPose:reconstruction/dex_retargeting/dex-retargeting:$PYTHONPATH
```

**`cannot make canonical path: meshes/visual/right_base_link.obj`**
The robot visual meshes have not been generated. Run
`python reconstruction/dex_retargeting/prepare_hand_meshes.py`, or just re-run
`setup_externals.sh`.

**`DRO-Grasp assets not found`**
Stage 8 needs the DRO-Grasp repo for robot URDFs and point clouds. Clone it next
to VideoManip or set `DRO_GRASP_ROOT=/path/to/DRO-Grasp`.

**Stage 2 hangs or has no window**
It is interactive by design. Use a machine with a display, or reuse cached click
prompts from `sam2/sam2/object_click.json`.

**A stage says `SKIP`**
That is deliberate — read its log; it names the missing key, weight, or checkout.

---

## 9. Notes

- Data, model weights, checkpoints, and downstream training/eval are intentionally
  excluded (see `.gitignore`); a fresh clone is ~3 MB of code + the example video
  + submodule pins.
- Robot-hand visual meshes are **generated**, not committed — `prepare_hand_meshes.py`
  converts the `.glb` meshes from the `dex-urdf` assets submodule to the `.obj`
  files our URDFs reference.
- API keys are read from the environment only. Never commit them.
- `grasping/` holds the optional ContactOpt grasp-refinement step. Downstream
  training/evaluation (grasping model, manipulation policy) is not in this repo.

---

See the [repository README](../README.md) for project status and the paper citation.
