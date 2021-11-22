#!/bin/bash

set -e +o pipefail
# set -x
# set -v

# Color variables
export b="\033[0;36m"
export g="\033[0;32m"
export r="\033[0;31m"
export e="\033[0;90m"
export y="\033[0;33m"
export x="\033[0m"

export TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:21.10-py3"}
export TRITON_URL=${TRITON_URL:-"localhost:8001"}

export USE_CPP=${USE_CPP:-1}

# RUN OPTIONS
export RUN_PYTORCH=${RUN_PYTORCH:-0}
export RUN_TRITON_ONNX=${RUN_TRITON_ONNX:-1}
export RUN_TRITON_XGB=${RUN_TRITON_XGB:-1}
export RUN_TRITON_TRT=${RUN_TRITON_TRT:-0}
export RUN_TENSORRT=${RUN_TENSORRT:-0}
