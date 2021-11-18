#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"morpheus"}
export DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"dev-$(date +'%y%m%d')"}
export DOCKER_TARGET=${DOCKER_TARGET:-"cudf_build"}

# Call the general build script
${SCRIPT_DIR}/build_container.sh
