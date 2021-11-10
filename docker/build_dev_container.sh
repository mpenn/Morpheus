#!/bin/bash

MORPHEUS_TAG=${MORPHEUS_TAG:-$(date +'%y%m%d')}
BUILD_TARGET=${BUILD_TARGET:-base}
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}

# Build args
FROM_IMAGE=${FROM_IMAGE:-"gpuci/miniconda-cuda"}
CUDA_VER=${CUDA_VER:-11.4}
LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}
LINUX_VER=${LINUX_VER:-20.04}
RAPIDS_VER=${RAPIDS_VER:-21.10}
PYTHON_VER=${PYTHON_VER:-3.8}
TENSORRT_VERSION=${TENSORRT_VERSION:-7.2.2.3}

echo "Building morpheus:${MORPHEUS_TAG}..."
echo "   FROM_IMAGE      : ${FROM_IMAGE}"
echo "   CUDA_VER        : ${CUDA_VER}"
echo "   LINUX_DISTRO    : ${LINUX_DISTRO}"
echo "   LINUX_VER       : ${LINUX_VER}"
echo "   RAPIDS_VER      : ${RAPIDS_VER}"
echo "   PYTHON_VER      : ${PYTHON_VER}"
echo "   TENSORRT_VERSION: ${TENSORRT_VERSION}"

# Export buildkit variable
export DOCKER_BUILDKIT

docker build -t morpheus:${MORPHEUS_TAG} --target ${BUILD_TARGET} --network=host \
   --build-arg FROM_IMAGE=${FROM_IMAGE} \
   --build-arg CUDA_VER=${CUDA_VER} \
   --build-arg LINUX_DISTRO=${LINUX_DISTRO} \
   --build-arg LINUX_VER=${LINUX_VER} \
   --build-arg RAPIDS_VER=${RAPIDS_VER} \
   --build-arg PYTHON_VER=${PYTHON_VER} \
   --build-arg TENSORRT_VERSION=${TENSORRT_VERSION} \
   -f docker/Dockerfile .
