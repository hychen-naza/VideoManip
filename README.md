# VideoManip

**Dexterous manipulation policies from RGB human videos, via 3D hand–object
trajectory reconstruction.**

Given an ordinary RGB video of a person manipulating an object, recover what
happened in 3D — the hand mesh, the object mesh, the object's 6-DoF pose over
time — and retarget it onto a robot hand, so the result can be used to train
manipulation policies.

This repository is the code release for the paper below. It is organized as
**our pipeline scripts** plus **thin wrappers** around several open-source
repos, which are vendored as pinned git submodules with a patch of our edits.

---

## Project status

| Component | Status | Where |
|---|---|---|
| **Reconstruction pipeline** — video → hand + object 3D trajectory → robot hand | ✅ **Done.** Runs end-to-end from a fresh clone; ships an example video and per-stage logging. | [`reconstruction/`](reconstruction/README.md) |
| **Grasping pipeline** — grasp dataset generation, DRO grasp model training and evaluation | 🚧 **Not yet released.** Only the optional ContactOpt grasp-refinement wrapper is here. | [`grasping/`](grasping/) |
| **Manipulation pipeline** — demo generation, 3D diffusion policy training, real-world evaluation | 🚧 **Not yet released.** | — |

> **Where things stand:** the reconstruction half of the paper is complete and
> reproducible — clone, set up, and run it on the bundled example video. The
> downstream halves (learning a grasping model, and training/evaluating a
> manipulation policy on the reconstructed trajectories) are not in this
> repository yet. The reconstruction outputs — per-frame MANO hand parameters,
> posed object meshes, and retargeted robot joint trajectories — are the
> hand-off point between them.

---

## Quick start

```bash
git clone --recursive https://github.com/hychen-naza/VideoManip.git
cd VideoManip

conda env create -f reconstruction/video_env.yml   # creates the `video` env
conda activate video
./reconstruction/setup_externals.sh                # patches submodules, builds hand meshes

cd reconstruction
./process_videos.sh                                # runs the bundled example video
```

Full instructions, a stage-by-stage walkthrough, and troubleshooting live in
**[`reconstruction/README.md`](reconstruction/README.md)**.

---

## Repository layout

```
reconstruction/     the reconstruction pipeline  -- start here
  README.md         setup, CLI, stage-by-stage walkthrough, troubleshooting
  EXTERNAL_REPOS.md how the submodules + our patches are organized
grasping/
  contactopt/       optional grasp / contact refinement (ContactOpt)
```

The pipeline builds on [SAM 2](https://github.com/facebookresearch/sam2),
[HaMeR](https://github.com/geopavlakos/hamer),
[MoGe](https://github.com/microsoft/MoGe),
[FoundationPose](https://github.com/NVlabs/FoundationPose),
[dex-retargeting](https://github.com/dexsuite/dex-retargeting) and
[ContactOpt](https://github.com/facebookresearch/ContactOpt). Each wrapper folder
documents how to install its component and what we changed.

---

## Citation

If you find this useful, please cite:

> Hongyi Chen, Tony Dong, Tiancheng Wu, Liquan Wang, Yash Jangir, Yaru Niu,
> Yufei Ye, Homanga Bharadhwaj, Zackory Erickson, Jeffrey Ichnowski.
> *Dexterous Manipulation Policies from RGB Human Videos via 3D Hand-Object
> Trajectory Reconstruction.* arXiv:2602.09013, 2026.

```bibtex
@article{chen2026dexterous,
  title   = {Dexterous Manipulation Policies from RGB Human Videos via 3D Hand-Object Trajectory Reconstruction},
  author  = {Chen, Hongyi and Dong, Tony and Wu, Tiancheng and Wang, Liquan and Jangir, Yash and Niu, Yaru and Ye, Yufei and Bharadhwaj, Homanga and Erickson, Zackory and Ichnowski, Jeffrey},
  journal = {arXiv preprint arXiv:2602.09013},
  year    = {2026}
}
```

Paper: https://arxiv.org/abs/2602.09013
