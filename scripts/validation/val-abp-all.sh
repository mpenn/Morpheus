#!/bin/bash

set -e +o pipefail
# set -x
# set -v

# RUN OPTIONS
# RUN_DOCKER=${RUN_DOCKER:-1}
RUN_PYTORCH=${RUN_PYTORCH:-0}
RUN_TRITON_XGB=${RUN_TRITON_XGB:-1}
RUN_TRITON_TRT=${RUN_TRITON_TRT:-0}
RUN_TENSORRT=${RUN_TENSORRT:-0}

TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:21.10-py3"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

INPUT_FILE=${INPUT_FILE:-"${MORPHEUS_ROOT}/data/nvsmi.jsonlines"}
TRUTH_FILE=${TRUTH_FILE:-"${MORPHEUS_ROOT}/data/nvsmi.jsonlines"}

MODEL_FILE=${MODEL_FILE:-"${MORPHEUS_ROOT}/models/abp-models/abp-${ABP_TYPE}-xgb-20210310.bst"}

# Now call the script to run
${SCRIPT_DIR}/val-abp.sh "nvsmi"
