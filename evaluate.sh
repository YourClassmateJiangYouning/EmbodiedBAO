#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./evaluate.sh MODEL_NAME [EPISODES]"
    exit 1
fi

MODEL="$1"
EPISODES="${2:-50}"

echo "Running EmbodiedBAO full evaluation for model: ${MODEL} (episodes=${EPISODES})"
"${ISAACSIM_ROOT}/python.sh" main.py --model "${MODEL}" --all-levels --episodes "${EPISODES}" --headless
