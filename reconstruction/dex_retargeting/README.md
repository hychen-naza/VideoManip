# dex_retargeting — hand→robot retargeting

[dex-retargeting](https://github.com/dexsuite/dex-retargeting) maps the recovered
MANO hand motion onto a robot hand (we use the **Inspire** hand).

```
dex_retargeting/
  dex-retargeting/   pinned submodule → dexsuite/dex-retargeting @ 8632b2c
                     (has its own nested submodules: assets=dex-urdf, manopth)
  dex_retargeting.patch   our edits to upstream (+69/−29 across 11 files)
  added/                  new files we authored
  README.md               (this file)
```

## Our changes
- **Patch** (`dex_retargeting.patch`) — `dex_retargeting/optimizer.py`,
  `constants.py`, `retargeting_config.py`, `seq_retarget.py`, the offline
  `inspire_hand_right.yml` / `leap_hand_right.yml` configs, and
  `example/position_retargeting/{dataset.py,hand_robot_viewer.py,mano_layer.py}`.
- **Added files** (`added/`, copied into `dex-retargeting/`):
  - `example/position_retargeting/retarget_hand_object_common_inspire.py` — pipeline entry (stage 8).
  - supporting scripts: `hand_robot_common.py`, `render_common.py`,
    `robot_hand_mesh.py`, `image2mesh_render*.py`, `retarget_*` and the
    `hand_utils/` package (optimization, multilateration, MANO fitting, …).
  - robot URDFs for `inspire/`, `leap_hand/`, `shadow_hand/` and
    `dex_retargeting/configs/offline/umi_gripper.yml`.

## How to add our code
```bash
# 1. fetch the submodule AND its nested submodules (dex-urdf assets, manopth)
git submodule update --init --recursive reconstruction/dex_retargeting/dex-retargeting

# 2. apply our patch
git -C reconstruction/dex_retargeting/dex-retargeting apply reconstruction/dex_retargeting/dex_retargeting.patch

# 3. copy our added files in
cp -r reconstruction/dex_retargeting/added/. reconstruction/dex_retargeting/dex-retargeting/

# 4. install + MANO
pip install -e reconstruction/dex_retargeting/dex-retargeting
#   MANO models (mano_v1_2) from https://mano.is.tue.mpg.de
```
> **Robot visual meshes** (`.obj/.glb/.dae`) under
> `example/position_retargeting/{inspire,leap_hand,shadow_hand}/meshes/` are **not**
> vendored (they duplicate the dex-urdf `assets` submodule). If a script reports a
> missing mesh, copy it from
> `reconstruction/dex_retargeting/dex-retargeting/assets/robots/hands/<hand>/meshes/`.

## Used by the pipeline
`process_videos.sh` runs, from `reconstruction/dex_retargeting/dex-retargeting/example/position_retargeting/`:
```bash
python retarget_hand_object_common_inspire.py \
       --objects <object> --use-optimize false            # stage 8
```
