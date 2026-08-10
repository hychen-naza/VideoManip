set -e  # Exit on any error

# This script runs INSIDE the foundationpose Docker container.
# Paths must therefore be valid in the container; run_object_pos_est_in_docker.sh
# bind-mounts the repo and forwards DATA_ROOT / CUDA_DEVICE.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directory for all object folders. Defaults to <repo>/reconstruction/data,
# matching process_videos.sh; override by exporting DATA_ROOT.
BASE_DIR="${DATA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)/data}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export DATA_ROOT="$BASE_DIR"
echo "Using data root: $BASE_DIR (GPU $CUDA_DEVICE)"

detect_obj_mesh_pose() {
    local object_folder="$1"
    local frames_dir="$BASE_DIR/$object_folder"

    if [ ! -d "$frames_dir" ]; then
        echo "Error: Directory $frames_dir does not exist!"
        return 1
    fi

    local rc=0
    for object_type in grasp target; do
        echo "--- FoundationPose: $object_folder ($object_type) ---"
        if ! CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python run_demo_multi_scale.py \
                --object_folder "$object_folder" --object_type "$object_type"; then
            # A scene may legitimately have no target object; report and keep going.
            echo "Warning: pose estimation failed for $object_folder ($object_type)"
            rc=1
        fi
    done
    return "$rc"
}


# loop through all objects passed as command-line arguments
# If no arguments provided, use default objects
if [ $# -eq 0 ]; then
    # Default objects if no arguments provided
    all_objects=('train_pan_8_1') # 'real_15_opendrawer'
else
    # Use command-line arguments as object list
    all_objects=("$@")
fi

for object_folder in "${all_objects[@]}"; do
    detect_obj_mesh_pose "$object_folder"
done