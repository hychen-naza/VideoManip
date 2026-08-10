#!/bin/bash
# Set up the vendored external repos for the VideoManip reconstruction pipeline.
#
# Each external repo is a *pinned git submodule* of its upstream, nested inside a
# wrapper folder that also holds our changes:
#   reconstruction/sam2/            { sam2/ (submodule) + sam2.patch + added/ }
#   reconstruction/hamer/           { hamer/ + hamer.patch + added/ }
#   reconstruction/dex_retargeting/ { dex-retargeting/ + dex_retargeting.patch + added/ }
#   grasping/contactopt/            { ContactOpt/ + contactopt.patch + added/ }
# (FoundationPose is set up separately via Docker — see reconstruction/foundationpose/README.md)
#
# Run this once after:  git clone --recursive <this repo>
# (or, if you forgot --recursive:  git submodule update --init --recursive)

# Note: no `set -e`. A component that fails to set up is reported and counted, so
# one run tells you everything that needs attention instead of stopping at the first.
set -uo pipefail
FAILURES=0

# Operate from the repository root so all paths below are repo-relative.
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

echo "==> Initializing submodules (upstream repos at pinned commits)..."
git submodule update --init --recursive

apply_pkg() {
  local wrapper="$1" repo="$2" patch="$3"   # wrapper dir, submodule path, patch filename
  echo "==> $wrapper"

  if [ ! -d "$ROOT/$repo/.git" ] && [ ! -f "$ROOT/$repo/.git" ]; then
    echo "    !! submodule $repo is not checked out; run:"
    echo "       git submodule update --init --recursive"
    return 1
  fi

  if [ -f "$ROOT/$wrapper/$patch" ] && [ -s "$ROOT/$wrapper/$patch" ]; then
    # Our patches are generated without --binary, so binary hunks (deleting an
    # upstream teaser image, etc.) can never apply. Ignore those paths and judge
    # success on the source hunks only.
    local apply_opts=(--whitespace=nowarn --exclude='*.jpg' --exclude='*.png'
                      --exclude='*.webp' --exclude='*.gif')
    if git -C "$ROOT/$repo" apply --check -R "${apply_opts[@]}" "$ROOT/$wrapper/$patch" 2>/dev/null; then
      echo "    $patch already applied - skipping"
    elif git -C "$ROOT/$repo" apply "${apply_opts[@]}" "$ROOT/$wrapper/$patch" 2>/dev/null; then
      echo "    applied $wrapper/$patch"
    else
      echo "    $patch did not apply as a whole; retrying hunk-by-hunk"
      if git -C "$ROOT/$repo" apply --reject "${apply_opts[@]}" "$ROOT/$wrapper/$patch"; then
        echo "    applied $wrapper/$patch (with --reject)"
      else
        local rejects
        rejects="$(find "$ROOT/$repo" -name '*.rej' 2>/dev/null | wc -l)"
        if [ "$rejects" -gt 0 ]; then
          echo "    !! $rejects hunk(s) rejected; inspect the .rej files under $repo" >&2
          FAILURES=$((FAILURES + 1))
        else
          # Nothing rejected means every source hunk was already present.
          echo "    $patch appears already applied"
        fi
      fi
    fi
  fi

  if [ -d "$ROOT/$wrapper/added" ]; then
    echo "    copying $wrapper/added/ -> $repo/"
    cp -r "$ROOT/$wrapper/added/." "$ROOT/$repo/"
  fi
}

apply_pkg reconstruction/sam2            reconstruction/sam2/sam2                       sam2.patch
apply_pkg reconstruction/hamer           reconstruction/hamer/hamer                     hamer.patch
apply_pkg reconstruction/dex_retargeting reconstruction/dex_retargeting/dex-retargeting dex_retargeting.patch
apply_pkg grasping/contactopt            grasping/contactopt/ContactOpt                 contactopt.patch

# The retargeting URDFs reference visual meshes as .obj, while the dex-urdf assets
# submodule ships them as .glb. Generate the .obj tree instead of vendoring ~13 MB.
echo "==> preparing robot hand meshes"
if command -v python >/dev/null 2>&1; then
  if python "$ROOT/reconstruction/dex_retargeting/prepare_hand_meshes.py"; then
    # Mirror them into the submodule copy the scripts actually run from.
    for hand in inspire leap_hand shadow_hand; do
      src="$ROOT/reconstruction/dex_retargeting/added/example/position_retargeting/$hand/meshes"
      dst="$ROOT/reconstruction/dex_retargeting/dex-retargeting/example/position_retargeting/$hand"
      [ -d "$src" ] && [ -d "$dst" ] && cp -r "$src" "$dst/"
    done
  else
    echo "    !! hand mesh preparation failed (needs trimesh in the active env)" >&2
    FAILURES=$((FAILURES + 1))
  fi
else
  echo "    !! python not found on PATH; skipping hand mesh preparation" >&2
  FAILURES=$((FAILURES + 1))
fi

cat <<'NOTE'

==> Done applying our patches + added files.

Manual steps still required (large / licensed assets we do NOT redistribute):

  * MANO models
      hamer and dex-retargeting need MANO (mano_v1_2). Download from
      https://mano.is.tue.mpg.de and place per each repo's README
      (e.g. reconstruction/hamer/hamer/_DATA/).

  * hamer weights + ViTPose
      Follow hamer's own README to fetch hamer_ckpts / vitpose_ckpts into
      reconstruction/hamer/hamer/_DATA and to install third-party/ViTPose.

  * FoundationPose
      See reconstruction/foundationpose/README.md (separate Docker-based setup).

  * DRO-Grasp (hand retargeting stage only)
      The retargeting stage loads robot URDFs and point clouds from the DRO-Grasp
      repo. Clone it next to VideoManip, or set DRO_GRASP_ROOT=/path/to/DRO-Grasp.

  * sam2 weights
      Download SAM2 checkpoints into reconstruction/sam2/sam2/checkpoints
      (cd reconstruction/sam2/sam2/checkpoints && ./download_ckpts.sh).
NOTE

if [ "$FAILURES" -gt 0 ]; then
  echo "==> setup_externals.sh finished with $FAILURES problem(s) - see the messages above." >&2
  exit 1
fi
echo "==> setup_externals.sh complete."
