#!/bin/bash
set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$SCRIPT_DIR"

# Path to the bash script to run inside docker
BASH_SCRIPT="$SCRIPT_DIR/object_pos_est.bash"

# Check if the bash script exists
if [ ! -f "$BASH_SCRIPT" ]; then
    echo "Error: Bash script not found at $BASH_SCRIPT"
    exit 1
fi

# Remove existing container if it exists
docker rm -f foundationpose 2>/dev/null || true

# Run docker container and execute the bash script inside it
# The script will run and the container will exit automatically when done
# Pass all command-line arguments to the bash script
# Build the command with all arguments properly quoted
CMD="cd $DIR && bash $BASH_SCRIPT"
for arg in "$@"; do
    CMD="$CMD \"$arg\""
done
echo "CMD: $CMD"
# Data root defaults to <repo>/reconstruction/data (same as process_videos.sh) and is
# forwarded into the container, which also needs it bind-mounted.
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)/data}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
echo "Data root: $DATA_ROOT"

docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -i --network=host --name foundationpose \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$DIR:$DIR" -v "$DATA_ROOT:$DATA_ROOT" -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp \
    --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE \
    -e DATA_ROOT="$DATA_ROOT" -e CUDA_DEVICE="$CUDA_DEVICE" \
    foundationpose:latest bash -c "$CMD"

# Clean up: remove the container after execution
docker rm -f foundationpose 2>/dev/null || true

echo "Script execution completed and container removed."

