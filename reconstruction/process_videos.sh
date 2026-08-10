#!/usr/bin/env bash
#
# VideoManip reconstruction pipeline driver.
#
# Runs the video -> hand/object reconstruction stages for one or more objects.
# Every path is resolved relative to this repository, so the script works from a
# fresh clone on any machine with no editing. Anything machine-specific is an
# environment variable or a command-line flag (see --help).
#
# Each stage runs in a subshell, is timed, and is logged to
#   $DATA_ROOT/<object>/logs/<NN>_<stage>.log
# A failing stage does NOT abort the run: it is recorded and the pipeline moves
# on, so a single invocation tells you everything that is broken. The exit code
# is the number of failed stages.
#
# Quick start (uses the bundled example video):
#   conda activate video
#   ./process_videos.sh
#
# See reconstruction/README.md for the full stage list and prerequisites.

set -uo pipefail

# ---------------------------------------------------------------- locations --
# Resolve this script's directory even when invoked via a symlink or from
# another cwd, without relying on git being present.
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR_="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR_/$SOURCE"
done
RECON_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
REPO_ROOT="$(cd -P "$RECON_DIR/.." && pwd)"

# ----------------------------------------------------------------- defaults --
# All overridable via environment or flags.
DATA_ROOT="${DATA_ROOT:-$RECON_DIR/data}"           # per-object working dirs
VIDEO_DIR="${VIDEO_DIR:-$RECON_DIR/example_data}"   # input videos / images
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FRAME_RATE="${FRAME_RATE:-1}"                       # keep every Nth frame
ROTATE_FRAMES="${ROTATE_FRAMES:-0}"                 # 1 = rotate 90deg clockwise
FOUNDATIONPOSE_DIR="${FOUNDATIONPOSE_DIR:-}"        # external, see README
STAGES=""                                           # empty = default set
OBJECTS=()

# Several packages (mmpose via ViTPose, dex_retargeting) are normally `pip install
# -e`'d. Those editable installs record an absolute path, so they break the moment
# the checkout is moved or cloned somewhere else. Put the in-repo copies first on
# PYTHONPATH so this clone is always what gets imported, installed or not.
for _p in \
    "$RECON_DIR/hamer/hamer/third-party/ViTPose" \
    "$RECON_DIR/dex_retargeting/dex-retargeting"
do
    [ -d "$_p" ] && export PYTHONPATH="$_p${PYTHONPATH:+:$PYTHONPATH}"
done
unset _p

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# All stages, in dependency order. Name:function.
ALL_STAGES=(
    "frames:generate_video_frames"
    "clicks:get_click_info"
    "intrinsics:generate_camera_intrinsics"
    "hand_mesh:detect_hand_mesh"
    "masks:generate_segmentation_masks"
    "obj_mesh:generate_obj_mesh"
    "obj_pose:detect_obj_mesh_pose"
    "retarget:retarget_hand"
)
# Stages run when --stages is not given. obj_pose needs Docker + an external
# FoundationPose checkout, so it is opt-in rather than default.
DEFAULT_STAGES="frames,clicks,intrinsics,hand_mesh,masks,obj_mesh,retarget"

# ------------------------------------------------------------------- usage ---
usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [object ...]

Reconstruct hand + object manipulation from RGB video(s).

Objects are looked up in \$VIDEO_DIR as <object>.<mp4|avi|png|jpg>. With no
object arguments, every video found in \$VIDEO_DIR is processed.

Options:
  --data-root DIR     Per-object working dirs   (default: $DATA_ROOT)
  --video-dir DIR     Input videos / images     (default: $VIDEO_DIR)
  --cuda-device N     GPU index                 (default: $CUDA_DEVICE)
  --frame-rate N      Keep every Nth frame      (default: $FRAME_RATE)
  --rotate            Rotate frames 90deg clockwise during extraction
  --stages LIST       Comma-separated stages to run
                      (default: $DEFAULT_STAGES)
  --list-stages       Print the stage names and exit
  -h, --help          Show this help and exit

Environment variables (same meaning as the flags): DATA_ROOT, VIDEO_DIR,
CUDA_DEVICE, FRAME_RATE, ROTATE_FRAMES, FOUNDATIONPOSE_DIR, MESHY_API_KEY,
OPENAI_API_KEY.

Examples:
  $(basename "$0")                             # bundled example video
  $(basename "$0") --stages frames,intrinsics real_14_pourtea
  VIDEO_DIR=~/my_videos $(basename "$0") my_object
EOF
}

list_stages() {
    echo "Stages in dependency order:"
    for entry in "${ALL_STAGES[@]}"; do
        printf '  %-12s %s\n' "${entry%%:*}" "${entry##*:}"
    done
    echo
    echo "Default: $DEFAULT_STAGES"
    echo "(obj_pose is opt-in: it needs Docker + an external FoundationPose checkout)"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --data-root)    DATA_ROOT="$2"; shift 2 ;;
        --video-dir)    VIDEO_DIR="$2"; shift 2 ;;
        --cuda-device)  CUDA_DEVICE="$2"; export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --frame-rate)   FRAME_RATE="$2"; shift 2 ;;
        --rotate)       ROTATE_FRAMES=1; shift ;;
        --stages)       STAGES="$2"; shift 2 ;;
        --list-stages)  list_stages; exit 0 ;;
        -h|--help)      usage; exit 0 ;;
        -*)             echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)              OBJECTS+=("$1"); shift ;;
    esac
done

[ -n "$STAGES" ] || STAGES="$DEFAULT_STAGES"

# Make the resolved paths visible to the Python stages, which read DATA_ROOT
# rather than hardcoding a location.
export DATA_ROOT VIDEO_DIR CUDA_DEVICE

# ------------------------------------------------------------------ logging --
C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BOLD=""
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_BOLD=$'\033[1m'
fi

log()      { printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "INFO " "$*"; }
log_warn() { printf '%s [%s] %s%s%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "WARN " "$C_YELLOW" "$*" "$C_RESET" >&2; }
log_err()  { printf '%s [%s] %s%s%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "ERROR" "$C_RED" "$*" "$C_RESET" >&2; }

# Stage bookkeeping, filled in by run_stage.
STAGE_NAMES=(); STAGE_STATUS=(); STAGE_SECONDS=(); STAGE_LOGS=()

# run_stage <index> <name> <function> <object>
# Runs <function> <object> in a subshell, tees output to a per-stage log, and
# records PASS / FAIL / SKIP. Never aborts the pipeline.
run_stage() {
    local idx="$1" name="$2" func="$3" object="$4"
    local log_dir="$DATA_ROOT/$object/logs"
    mkdir -p "$log_dir"
    local log_file
    log_file="$(printf '%s/%02d_%s.log' "$log_dir" "$idx" "$name")"

    echo
    log "${C_BOLD}--- stage $idx: $name ---${C_RESET}  (log: $log_file)"
    local start end rc
    start=$(date +%s)
    # Subshell: a stage that cd's cannot affect the next stage.
    ( set -o pipefail; "$func" "$object" ) 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    end=$(date +%s)

    local status
    case "$rc" in
        0)  status="PASS"; log "stage $name: ${C_GREEN}PASS${C_RESET} ($((end-start))s)" ;;
        # 3 is our convention for "prerequisite missing, deliberately skipped".
        3)  status="SKIP"; log_warn "stage $name: SKIP ($((end-start))s) - see $log_file" ;;
        *)  status="FAIL"; log_err "stage $name: FAIL rc=$rc ($((end-start))s) - see $log_file" ;;
    esac
    STAGE_NAMES+=("$name"); STAGE_STATUS+=("$status")
    STAGE_SECONDS+=("$((end-start))"); STAGE_LOGS+=("$log_file")
}

# require_dir <path> <human description> -- returns 1 (fail) when missing.
require_dir() {
    if [ ! -d "$1" ]; then
        log_err "missing $2: $1"
        return 1
    fi
}

# require_frames <object> -- every stage after stage 1 needs extracted frames.
require_frames() {
    local rgb_dir="$DATA_ROOT/$1/rgb"
    if [ ! -d "$rgb_dir" ] || [ -z "$(ls -A "$rgb_dir" 2>/dev/null)" ]; then
        log_err "no extracted frames in $rgb_dir - run the 'frames' stage first"
        return 1
    fi
}

# ------------------------------------------------------------------ stages ---

# Stage 1: video (or single image) -> RGB frames in <object>/rgb/
generate_video_frames() {
    local object="$1"
    local file_path
    file_path="$(find "$VIDEO_DIR" -maxdepth 1 -name "$object.*" -type f 2>/dev/null | sort | head -n 1)"

    if [ -z "$file_path" ]; then
        log_err "no input file named '$object.*' in $VIDEO_DIR"
        return 1
    fi
    log "input: $file_path"

    local ext="${file_path##*.}"
    local rgb_dir="$DATA_ROOT/$object/rgb"

    case "${ext,,}" in
        png|jpg|jpeg)
            mkdir -p "$rgb_dir"
            cp "$file_path" "$rgb_dir/" || return 1
            log "copied single image into $rgb_dir"
            ;;
        mp4|avi|mov|mkv)
            local rotate_flag=()
            [ "$ROTATE_FRAMES" = "1" ] && rotate_flag=(--rotate)
            python "$RECON_DIR/utils/video_parser.py" "$file_path" \
                -o "$rgb_dir" -r "$FRAME_RATE" "${rotate_flag[@]}" || return 1
            ;;
        *)
            log_err "unsupported input type '.$ext' for $file_path"
            return 1
            ;;
    esac

    local n
    n="$(find "$rgb_dir" -name '*.png' -o -name '*.jpg' | wc -l)"
    log "extracted $n frames into $rgb_dir"
    [ "$n" -gt 0 ] || { log_err "no frames were written"; return 1; }
}

# Stage 2: object click prompts for SAM 2.
# This is an INTERACTIVE matplotlib step. If prompts for this object were
# already saved, it is a no-op; if no display is available it skips with
# instructions rather than hanging forever.
get_click_info() {
    local object="$1"
    require_frames "$object" || return 1

    local click_json="$RECON_DIR/sam2/sam2/object_click.json"
    if [ ! -f "$click_json" ]; then
        log_err "click database not found: $click_json"
        log_err "run reconstruction/setup_externals.sh to copy sam2/added/ into the submodule"
        return 1
    fi

    if python - "$click_json" "$object" <<'PY'
import json, sys
path, obj = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
entry = data.get(obj)
ok = bool(entry) and any(e.get("clicked_points") for e in entry)
print(f"saved click prompts for '{obj}': {'yes' if ok else 'no'}")
sys.exit(0 if ok else 1)
PY
    then
        log "reusing saved click prompts - nothing to do"
        return 0
    fi

    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        log_warn "no click prompts saved for '$object' and no display is available."
        log_warn "This stage needs an interactive window. On a machine with a GUI run:"
        log_warn "  cd $RECON_DIR/sam2/sam2 && python get_click_info.py $DATA_ROOT/$object/rgb"
        log_warn "then re-run this pipeline. Skipping."
        return 3
    fi

    ( cd "$RECON_DIR/sam2/sam2" && python get_click_info.py "$DATA_ROOT/$object/rgb" )
}

# Stage 3: MoGe depth + camera intrinsics -> <object>/cam_info/, <object>/depth/
generate_camera_intrinsics() {
    local object="$1"
    require_frames "$object" || return 1
    python "$RECON_DIR/utils/moge_depth.py" "$DATA_ROOT/$object/rgb" \
        --output_folder "$DATA_ROOT/$object/cam_info" || return 1
    [ -f "$DATA_ROOT/$object/cam_K.txt" ] || { log_err "cam_K.txt was not written"; return 1; }
}

# Stage 4: HaMeR hand mesh + MANO params -> <object>/human_hand/
detect_hand_mesh() {
    local object="$1"
    require_frames "$object" || return 1

    if [ ! -f "$DATA_ROOT/$object/cam_K.txt" ]; then
        log_err "cam_K.txt missing - run the 'intrinsics' stage first"
        return 1
    fi
    if [ ! -d "$RECON_DIR/hamer/hamer/_DATA" ]; then
        log_warn "hamer weights not found at reconstruction/hamer/hamer/_DATA"
        log_warn "see reconstruction/hamer/README.md to download hamer_ckpts / vitpose_ckpts / MANO"
        return 3
    fi

    mkdir -p "$DATA_ROOT/$object/human_hand"
    ( cd "$RECON_DIR/hamer/hamer" && python hand_detection.py \
        --img_folder "$DATA_ROOT/$object/rgb" \
        --out_folder "$DATA_ROOT/$object/human_hand" )
}

# Stage 5: SAM 2 object masks -> <object>/masks_pred_obj/ + croped_frames/
generate_segmentation_masks() {
    local object="$1"
    require_frames "$object" || return 1

    local ckpt="$RECON_DIR/sam2/sam2/checkpoints/sam2.1_hiera_large.pt"
    if [ ! -e "$ckpt" ]; then
        log_warn "SAM 2 checkpoint missing: $ckpt"
        log_warn "download it with: cd $RECON_DIR/sam2/sam2/checkpoints && ./download_ckpts.sh"
        return 3
    fi

    mkdir -p "$DATA_ROOT/$object/masks_pred_obj" "$DATA_ROOT/$object/masks_pred_hand"
    ( cd "$RECON_DIR/sam2/sam2" && python run_video_function.py "$DATA_ROOT/$object/rgb" \
        --hand_input_folder "$DATA_ROOT/$object/human_hand" \
        --obj_output_dir "$DATA_ROOT/$object/masks_pred_obj" \
        --hand_output_dir "$DATA_ROOT/$object/masks_pred_hand" )
}

# Stage 6: cropped object image -> textured mesh (Meshy + OpenAI APIs).
generate_obj_mesh() {
    local object="$1"
    local croped_dir="$DATA_ROOT/$object/croped_frames"

    if [ ! -d "$croped_dir" ] || [ -z "$(ls -A "$croped_dir" 2>/dev/null)" ]; then
        log_err "no cropped frames in $croped_dir - run the 'masks' stage first"
        return 1
    fi
    if [ -z "${MESHY_API_KEY:-}" ] || [ -z "${OPENAI_API_KEY:-}" ]; then
        log_warn "MESHY_API_KEY and/or OPENAI_API_KEY are not set."
        log_warn "This stage calls the Meshy image-to-3D API and the OpenAI API, both paid."
        log_warn "Export both keys to run it, or supply your own mesh at"
        log_warn "  $DATA_ROOT/$object/mesh/"
        return 3
    fi

    python "$RECON_DIR/utils/img2mesh.py" --image_dir "$croped_dir"
}

# Stage 7: FoundationPose 6-DoF object pose (external checkout, Docker).
detect_obj_mesh_pose() {
    local object="$1"
    require_frames "$object" || return 1

    if [ -z "$FOUNDATIONPOSE_DIR" ] || [ ! -d "$FOUNDATIONPOSE_DIR" ]; then
        log_warn "FOUNDATIONPOSE_DIR is not set to a FoundationPose checkout."
        log_warn "FoundationPose is not vendored here; see reconstruction/foundationpose/README.md,"
        log_warn "then re-run with FOUNDATIONPOSE_DIR=/path/to/FoundationPose --stages obj_pose"
        return 3
    fi
    if ! command -v docker >/dev/null 2>&1; then
        log_warn "docker not found on PATH; this stage runs inside the foundationpose image"
        return 3
    fi

    log "launching FoundationPose pose estimation for '$object'"
    "$RECON_DIR/foundationpose/scripts/run_object_pos_est_in_docker.sh" "$object"
}

# Stage 8: retarget the human hand onto a robot hand.
retarget_hand() {
    local object="$1"
    local use_optimize="${USE_OPTIMIZE:-false}"

    if [ ! -d "$DATA_ROOT/$object/human_hand" ]; then
        log_err "no human_hand/ output - run the 'hand_mesh' stage first"
        return 1
    fi
    local retarget_dir="$RECON_DIR/dex_retargeting/dex-retargeting/example/position_retargeting"
    if [ ! -d "$retarget_dir" ]; then
        log_warn "dex-retargeting not set up; run reconstruction/setup_externals.sh"
        return 3
    fi

    ( cd "$retarget_dir" && python retarget_hand_object_common_inspire.py \
        --objects "$object" --use-optimize "$use_optimize" )
}

# -------------------------------------------------------------- object loop --
process_object() {
    local object="$1"
    echo
    log "${C_BOLD}==========================================${C_RESET}"
    log "${C_BOLD}Object: $object${C_RESET}"
    log "  data root : $DATA_ROOT/$object"
    log "  video dir : $VIDEO_DIR"
    log "  GPU       : $CUDA_DEVICE"
    log "  stages    : $STAGES"
    log "${C_BOLD}==========================================${C_RESET}"

    mkdir -p "$DATA_ROOT/$object"

    STAGE_NAMES=(); STAGE_STATUS=(); STAGE_SECONDS=(); STAGE_LOGS=()
    local idx=0
    for entry in "${ALL_STAGES[@]}"; do
        local name="${entry%%:*}" func="${entry##*:}"
        idx=$((idx+1))
        case ",$STAGES," in
            *",$name,"*) run_stage "$idx" "$name" "$func" "$object" ;;
            *) ;;  # not selected
        esac
    done

    # ---- summary ----
    echo
    log "${C_BOLD}Summary for $object${C_RESET}"
    printf '  %-12s %-6s %8s  %s\n' "STAGE" "STATUS" "TIME" "LOG"
    local failed=0 i
    for i in "${!STAGE_NAMES[@]}"; do
        local color=""
        case "${STAGE_STATUS[$i]}" in
            PASS) color="$C_GREEN" ;;
            SKIP) color="$C_YELLOW" ;;
            FAIL) color="$C_RED"; failed=$((failed+1)) ;;
        esac
        printf '  %-12s %s%-6s%s %7ss  %s\n' \
            "${STAGE_NAMES[$i]}" "$color" "${STAGE_STATUS[$i]}" "$C_RESET" \
            "${STAGE_SECONDS[$i]}" "${STAGE_LOGS[$i]}"
    done
    echo
    return "$failed"
}

# --------------------------------------------------------------------- main --
if [ ! -d "$VIDEO_DIR" ]; then
    log_err "video directory does not exist: $VIDEO_DIR"
    log_err "pass --video-dir DIR or set VIDEO_DIR"
    exit 2
fi

# No objects given: derive them from the files in VIDEO_DIR.
if [ ${#OBJECTS[@]} -eq 0 ]; then
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        b="$(basename "$f")"
        OBJECTS+=("${b%.*}")
    done < <(find "$VIDEO_DIR" -maxdepth 1 -type f \
                \( -name '*.mp4' -o -name '*.avi' -o -name '*.mov' -o -name '*.mkv' \
                   -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | sort)
    if [ ${#OBJECTS[@]} -eq 0 ]; then
        log_err "no video or image files found in $VIDEO_DIR"
        exit 2
    fi
    log "no objects given; found ${#OBJECTS[@]} input(s) in $VIDEO_DIR: ${OBJECTS[*]}"
fi

mkdir -p "$DATA_ROOT"
TOTAL_FAILED=0
for object in "${OBJECTS[@]}"; do
    process_object "$object" || TOTAL_FAILED=$((TOTAL_FAILED + $?))
done

if [ "$TOTAL_FAILED" -gt 0 ]; then
    log_err "$TOTAL_FAILED stage(s) failed - inspect the logs listed above"
else
    log "${C_GREEN}all selected stages completed${C_RESET}"
fi
exit "$TOTAL_FAILED"
