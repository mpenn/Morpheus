#!/bin/bash

MORPHEUS_TAG=${MORPHEUS_TAG:-$(date +'%y%m%d')}

DOCKER_ARGS="-v $PWD:/workspace --net=host --gpus=all"

if [[ -z "${SSH_AUTH_SOCK}" ]]; then
   echo "No ssh-agent auth socket found. Dependencies in private git repos may fail during build."
else
   echo "Setting up ssh-agent auth socket"
   DOCKER_ARGS="${DOCKER_ARGS} -v ${SSH_AUTH_SOCK}:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent"
fi

echo "Launching morpheus:${MORPHEUS_TAG}..."

docker run --rm -ti ${DOCKER_ARGS} morpheus:${MORPHEUS_TAG} bash
