#!/bin/bash

set -e +o pipefail
# set -x
# set -v


# RUN OPTIONS
RUN_PYTORCH=${RUN_PYTORCH:-1}
RUN_TRITON_ONNX=${RUN_TRITON_ONNX:-1}
RUN_TRITON_TRT=${RUN_TRITON_TRT:-1}
RUN_TENSORRT=${RUN_TENSORRT:-0}

TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:21.10-py3"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MORPHEUS_ROOT=$(realpath ${MORPHEUS_ROOT:-"${SCRIPT_DIR}/../.."})

INPUT_FILE=${INPUT_FILE:-"${MORPHEUS_ROOT}/data/sid_training_data_truth.csv"}
TRUTH_FILE=${TRUTH_FILE:-"${MORPHEUS_ROOT}/data/sid_training_data_truth.csv"}

# Call minibert first
${SCRIPT_DIR}/val-sid.sh "minibert"
${SCRIPT_DIR}/val-sid.sh "bert"
