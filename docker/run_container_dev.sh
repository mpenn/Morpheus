#!/bin/bash

# Color variables
b="\033[0;36m"
g="\033[0;32m"
r="\033[0;31m"
e="\033[0;90m"
y="\033[0;33m"
x="\033[0m"

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"morpheus"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"dev-$(date +'%y%m%d')"}

DOCKER_ARGS="-v $PWD:/workspace --net=host --gpus=all"

if [[ -z "${SSH_AUTH_SOCK}" ]]; then
   echo -e "${y}No ssh-agent auth socket found. Dependencies in private git repos may fail during build.${x}"
else
   echo -e "${b}Setting up ssh-agent auth socket${x}"
   DOCKER_ARGS="${DOCKER_ARGS} -v $(readlink -f $SSH_AUTH_SOCK):/ssh-agent:ro -e SSH_AUTH_SOCK=/ssh-agent"
fi

echo -e "${g}Launching ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}...${x}"

docker run --rm -ti ${DOCKER_ARGS} ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} bash
