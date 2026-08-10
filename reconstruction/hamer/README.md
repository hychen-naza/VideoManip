# hamer — HaMeR hand mesh recovery

[HaMeR](https://github.com/geopavlakos/hamer) recovers a **MANO hand mesh** per
frame, used as the human-hand geometry for retargeting.

```
hamer/
  hamer/           pinned submodule → geopavlakos/hamer @ 091de2a
  hamer.patch      our edits to upstream (+685/−109 across 9 files)
  added/           new files we authored
  README.md        (this file)
```

## Our changes
- **Patch** (`hamer.patch`) — model + rendering changes:
  `hamer/models/hamer.py` (+291), `hamer/utils/renderer.py` (+370),
  `demo.py` (+109), `hamer/models/__init__.py`, `datasets/vitdet_dataset.py`,
  `models/heads/mano_head.py`, `utils/geometry.py`, `setup.py` (removes the
  vendored `assets/teaser.jpg`).
- **Added files** (`added/`, copied into `hamer/`):
  - `hand_detection.py` — pipeline entry: detect + fit the hand mesh (stage 4).
  - `manotorch_test.py`, `test_mano_conversion.py` — MANO conversion checks.

## How to add our code
```bash
# 1. fetch the pinned upstream submodule (and its nested deps)
git submodule update --init --recursive reconstruction/hamer/hamer

# 2. apply our patch
git -C reconstruction/hamer/hamer apply reconstruction/hamer/hamer.patch

# 3. copy our added files in
cp -r reconstruction/hamer/added/. reconstruction/hamer/hamer/

# 4. install HaMeR + fetch weights & ViTPose (follow hamer's own README)
pip install -e reconstruction/hamer/hamer
#   - third-party/ViTPose  : install per hamer README (hamer's own dependency)
#   - reconstruction/hamer/hamer/_DATA/hamer_ckpts, _DATA/vitpose_ckpts : download
#   - MANO models (mano_v1_2) from https://mano.is.tue.mpg.de -> _DATA/
```
> The large `_DATA/` weights, `third-party/ViTPose`, and MANO models are **not**
> redistributed here — download them as above.

## Used by the pipeline
`process_videos.sh` runs, from `reconstruction/hamer/hamer/`:
```bash
python hand_detection.py \
       --img_folder  <demo_data>/<object>/rgb \
       --out_folder  <demo_data>/<object>/human_hand          # stage 4
```
