#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"nvcr.io/ea-nvidia-morpheus/morpheus-sdk-cli"}
export DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"latest"}
export DOCKER_TARGET=${DOCKER_TARGET:-"runtime"}

# Call the general build script
${SCRIPT_DIR}/build_container.sh
