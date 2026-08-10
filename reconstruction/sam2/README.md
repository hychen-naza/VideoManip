# sam2 — SAM 2 for the reconstruction pipeline

[SAM 2](https://github.com/facebookresearch/sam2) is used for **interactive
click prompting** and **video object/hand mask propagation**.

```
sam2/
  sam2/            pinned submodule → facebookresearch/sam2 @ 2b90b9f
  sam2.patch       our edit to upstream (utils/misc.py, +5/−4)
  added/           new files we authored
  README.md        (this file)
```

## Our changes
- **Patch** (`sam2.patch`) — `sam2/utils/misc.py` (+5/−4): frame-loading tweak.
- **Added files** (`added/`, copied into `sam2/`):
  - `get_click_info.py` — collect the click prompts for each object (stage 2).
  - `run_video_function.py` — propagate object + hand masks through the clip (stage 5).
  - `run_video.py` — standalone variant.
  - `object_click.json` — saved click coordinates.

## How to add our code
From the repo root, either run `./setup_externals.sh` (does all submodules), or
just this one manually:

```bash
# 1. fetch the pinned upstream submodule
git submodule update --init reconstruction/sam2/sam2

# 2. apply our patch (applies cleanly — submodule is pinned to the patch's base)
git -C reconstruction/sam2/sam2 apply reconstruction/sam2/sam2.patch

# 3. copy our added files into the submodule
cp -r reconstruction/sam2/added/. reconstruction/sam2/sam2/

# 4. install SAM 2 + download checkpoints (see sam2's own README)
pip install -e reconstruction/sam2/sam2
#   put SAM2 checkpoints in reconstruction/sam2/sam2/checkpoints/
```

## Used by the pipeline
`process_videos.sh` runs, from `reconstruction/sam2/sam2/`:
```bash
python get_click_info.py      <demo_data>/<object>/rgb            # stage 2
python run_video_function.py  <demo_data>/<object>/rgb \
       --hand_input_folder <...>/human_hand \
       --obj_output_dir  <...>/masks_pred_obj \
       --hand_output_dir <...>/masks_pred_hand                    # stage 5
```
