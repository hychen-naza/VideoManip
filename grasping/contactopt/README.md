# contactopt — ContactOpt hand–object refinement

[ContactOpt](https://github.com/facebookresearch/ContactOpt) refines the hand and
object grasp so their contact is physically consistent. It is an **optional**
post-processing step (commented out by default in `process_videos.sh`).

```
contactopt/
  ContactOpt/          pinned submodule → facebookresearch/ContactOpt @ 9eeb59a
  contactopt.patch     our edits to upstream (+132/−64 across 6 files)
  added/               new files we authored
  README.md            (this file)
```

## Our changes
- **Patch** (`contactopt.patch`) — `contactopt/run_user_demo.py` (pipeline entry),
  `hand_object.py`, `optimize_pose.py`, `pointnet.py`, `run_contactopt.py`,
  `util.py`.
- **Added files** (`added/`, copied into `ContactOpt/`):
  - `setup.py` — makes the package installable.

## How to add our code
```bash
# 1. fetch the pinned upstream submodule
git submodule update --init grasping/contactopt/ContactOpt

# 2. apply our patch
git -C grasping/contactopt/ContactOpt apply grasping/contactopt/contactopt.patch

# 3. copy our added files in
cp -r grasping/contactopt/added/. grasping/contactopt/ContactOpt/

# 4. install + assets (see ContactOpt's own README)
pip install -e grasping/contactopt/ContactOpt
#   MANO models + ContactOpt's pretrained weights as per upstream README
```

## Used by the pipeline
`process_videos.sh` (enable by uncommenting `run_contactopt`), from
`grasping/contactopt/ContactOpt/`:
```bash
python ./contactopt/run_user_demo.py --objects <object>      # optional post-proc
```
