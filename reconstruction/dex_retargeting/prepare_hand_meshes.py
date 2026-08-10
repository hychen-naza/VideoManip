#!/usr/bin/env python3
"""
Materialize the robot-hand meshes referenced by our retargeting URDFs.

Our URDFs under
    reconstruction/dex_retargeting/added/example/position_retargeting/<hand>/
reference `meshes/visual/*.obj` and `meshes/collision/*.obj`, but the vendored
dex-urdf assets submodule ships the *visual* meshes as `.glb`. Rather than
committing ~13 MB of duplicated binaries, this script builds the `meshes/` tree
from the assets submodule:

    collision/*.obj   copied as-is
    visual/*.glb  ->  visual/*.obj   (converted with trimesh)

It is idempotent: existing up-to-date outputs are left alone. Run it once after
setup_externals.sh (that script calls it automatically).

Usage:
    python prepare_hand_meshes.py [--hands inspire_hand leap_hand] [--force]
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMODULE = HERE / "dex-retargeting"
ASSETS = SUBMODULE / "assets" / "robots" / "hands"

# Maps the dex-urdf asset folder -> the wrapper folder our URDFs live in.
HAND_DIRS = {
    "inspire_hand": "inspire",
    "leap_hand": "leap_hand",
    "shadow_hand": "shadow_hand",
}


def convert_glb_to_obj(src: Path, dst: Path) -> bool:
    """Convert a .glb mesh to .obj. Returns True on success."""
    try:
        import trimesh
    except ImportError:
        print("ERROR: trimesh is required to convert .glb meshes to .obj "
              "(pip install trimesh)", file=sys.stderr)
        raise

    try:
        loaded = trimesh.load(str(src), force="mesh", process=False)
    except Exception as exc:  # noqa: BLE001 - report and continue with the rest
        print(f"  !! failed to load {src.name}: {exc}", file=sys.stderr)
        return False

    if loaded is None or not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
        print(f"  !! {src.name} produced an empty mesh; skipping", file=sys.stderr)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        loaded.export(str(dst))
    except Exception as exc:  # noqa: BLE001
        print(f"  !! failed to write {dst.name}: {exc}", file=sys.stderr)
        return False
    return True


def prepare_hand(asset_name: str, wrapper_name: str, force: bool) -> int:
    """Build <wrapper>/meshes for one hand. Returns the number of failures."""
    src_meshes = ASSETS / asset_name / "meshes"
    dst_meshes = HERE / "added" / "example" / "position_retargeting" / wrapper_name / "meshes"

    if not src_meshes.is_dir():
        print(f"==> {wrapper_name}: no assets at {src_meshes} - skipping")
        print("    (run: git submodule update --init --recursive)")
        return 0

    print(f"==> {wrapper_name}: {src_meshes} -> {dst_meshes}")
    failures = 0

    # 1. collision meshes are already .obj -- copy verbatim.
    n_copied = 0
    for src in sorted((src_meshes / "collision").glob("*.obj")):
        dst = dst_meshes / "collision" / src.name
        if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    # 2. visual meshes ship as .glb -- convert to .obj (+ .mtl written alongside).
    n_converted = 0
    for src in sorted((src_meshes / "visual").glob("*.glb")):
        dst = dst_meshes / "visual" / (src.stem + ".obj")
        if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        if convert_glb_to_obj(src, dst):
            n_converted += 1
        else:
            failures += 1

    # Some visual meshes may already be .obj upstream; copy those too.
    for src in sorted((src_meshes / "visual").glob("*.obj")):
        dst = dst_meshes / "visual" / src.name
        if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    print(f"    copied {n_copied} collision/visual .obj, converted {n_converted} .glb")
    if failures:
        print(f"    !! {failures} mesh(es) failed to convert", file=sys.stderr)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", nargs="+", default=list(HAND_DIRS),
                        choices=list(HAND_DIRS),
                        help="Which hands to prepare (default: all known)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even when the output looks up to date")
    args = parser.parse_args()

    if not SUBMODULE.is_dir():
        print(f"ERROR: dex-retargeting submodule not found at {SUBMODULE}", file=sys.stderr)
        print("Run: git submodule update --init --recursive", file=sys.stderr)
        return 1

    failures = sum(prepare_hand(h, HAND_DIRS[h], args.force) for h in args.hands)
    if failures:
        print(f"\n{failures} mesh(es) could not be prepared", file=sys.stderr)
        return 1
    print("\nHand meshes ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
