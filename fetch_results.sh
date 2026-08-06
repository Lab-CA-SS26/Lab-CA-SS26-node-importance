#!/bin/bash
set -e

# Ensure we are in the root directory of the project locally
cd "$(dirname "$0")"

# SSH Configuration
REMOTE_HOST="coan-wrk-01"

# NOTE: Update this path if the project folder is named differently on the server!
REMOTE_PATH="~/Lab-CA-SS26-node-importance/benchmark/output/"
LOCAL_PATH="benchmark/output/"

echo "======================================"
echo "    Fetching Benchmark Results        "
echo "======================================"
echo "Connecting to $REMOTE_HOST..."

# Use rsync to efficiently sync the output directory.
# -a: archive mode (preserves permissions, recursive)
# -v: verbose
# -z: compress file data during transfer
# --progress: show progress bar
rsync -avz --progress "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"

echo "======================================"
echo "    Sync complete!                    "
echo "    You can now run:                  "
echo "    bash run_pipeline.sh -n           "
echo "======================================"
