#!/bin/bash
set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if a command/script was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command_or_script> [args...]"
    echo "Example: $0 bash /path/to/script.sh"
    echo "Example: $0 python script.py --arg1 value1"
    exit 1
fi

# Remove existing container if it exists
docker rm -f foundationpose 2>/dev/null || true

# Build the command to run inside docker
# If first argument is a file path, make it absolute
CMD="$@"
if [ -f "$1" ]; then
    # If it's a file, use absolute path
    ABS_PATH="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    CMD="${ABS_PATH}${CMD#$1}"
fi

# Run docker container and execute the command inside it
# Use -i (interactive) to allow input, but remove -t (tty) so it exits automatically
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -i --network=host --name foundationpose \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$DIR:$DIR" -v /home:/home -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp \
    --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE \
    foundationpose:latest bash -c "cd $DIR && $CMD"

# Clean up: remove the container after execution
docker rm -f foundationpose 2>/dev/null || true

echo "Command execution completed and container removed."

