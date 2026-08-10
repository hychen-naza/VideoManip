# External repositories

The reconstruction pipeline builds on several open-source repos. To keep this
repo small and give proper upstream attribution, we **do not vendor their code**.
Instead each one is a **pinned git submodule** of its upstream, nested **inside**
a wrapper folder together with our patch and added files. The wrappers live under
`reconstruction/` (except ContactOpt, which lives under `grasping/`):

```
reconstruction/sam2/
  sam2/          <- pinned submodule (facebookresearch/sam2 @ 2b90b9f)
  sam2.patch     <- our edits to upstream tracked files
  added/         <- new files we authored (copied into sam2/ by setup)
```

| Wrapper folder | Submodule (inside it) | Upstream | Pinned | Our changes |
|----------------|-----------------------|----------|--------|-------------|
| `reconstruction/sam2/` | `reconstruction/sam2/sam2` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | `2b90b9f` | tiny `misc.py` patch (+5/−4) + `get_click_info.py`, `run_video_function.py` |
| `reconstruction/hamer/` | `reconstruction/hamer/hamer` | [geopavlakos/hamer](https://github.com/geopavlakos/hamer) | `091de2a` | 685-line patch (model/renderer/demo) + `hand_detection.py` |
| `reconstruction/dex_retargeting/` | `reconstruction/dex_retargeting/dex-retargeting` | [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | `8632b2c` | patch + our `example/position_retargeting/` scripts |
| `grasping/contactopt/` | `grasping/contactopt/ContactOpt` | [facebookresearch/ContactOpt](https://github.com/facebookresearch/ContactOpt) | `9eeb59a` | patch + `setup.py` |
| `reconstruction/foundationpose/` | *(not a submodule — Docker)* | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | `e3d597b` | see its own README |

> FoundationPose is set up separately (Docker); see
> `reconstruction/foundationpose/README.md`. It is not wired as a submodule
> because it runs inside its own container.

## Quick start

```bash
git clone --recursive https://github.com/hychen-naza/VideoManip.git
cd VideoManip
./reconstruction/setup_externals.sh   # inits submodules, applies patches, copies our files
```

If you cloned without `--recursive`:
```bash
git submodule update --init --recursive
./reconstruction/setup_externals.sh
```
(`setup_externals.sh` finds the repo root itself, so it works from anywhere.)

## How each wrapper folder is structured
- `<repo>/` — the pinned upstream submodule.
- `*.patch` — `git diff` of our edits to upstream *tracked* files. Because the
  submodule is pinned to the exact commit the patch was made against, it always
  applies cleanly (e.g. `git -C reconstruction/sam2/sam2 apply reconstruction/sam2/sam2.patch`).
- `added/` — new files we authored, copied into the submodule by the setup script.

## Not included (download separately)
Model weights and licensed assets are **not** redistributed here:
- MANO models (hamer, dex-retargeting) — https://mano.is.tue.mpg.de
- hamer / ViTPose checkpoints — hamer README
- SAM2 checkpoints — sam2 README
- FoundationPose weights — FoundationPose README
- Large robot visual meshes for
  `reconstruction/dex_retargeting/dex-retargeting/example/position_retargeting/*`
  come from the `reconstruction/dex_retargeting/dex-retargeting/assets` (dex-urdf)
  submodule.
