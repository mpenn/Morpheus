#!/bin/bash

set -e +o pipefail
# set -x
# set -v

# Color variables
b="\033[0;36m"
g="\033[0;32m"
r="\033[0;31m"
e="\033[0;90m"
y="\033[0;33m"
x="\033[0m"

# RUN OPTIONS
# RUN_DOCKER=${RUN_DOCKER:-1}
RUN_PYTORCH=${RUN_PYTORCH:-0}
RUN_TRITON_XGB=${RUN_TRITON_XGB:-1}
RUN_TRITON_TRT=${RUN_TRITON_TRT:-0}
RUN_TENSORRT=${RUN_TENSORRT:-0}

TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:21.10-py3"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MORPHEUS_ROOT=$(realpath ${MORPHEUS_ROOT:-"${SCRIPT_DIR}/../.."})

INPUT_FILE=${INPUT_FILE:-"${MORPHEUS_ROOT}/data/nvsmi.jsonlines"}
TRUTH_FILE=${TRUTH_FILE:-"${MORPHEUS_ROOT}/data/nvsmi.jsonlines"}

# Get the ABP_MODEL from the argument. Must be 'nvsmi' for now
ABP_TYPE=${ABP_TYPE:-$1}

MODEL_FILE=${MODEL_FILE:-"${MORPHEUS_ROOT}/models/abp-models/abp-${ABP_TYPE}-xgb-20210310.bst"}
MODEL_DIRECTORY=${MODEL_FILE%/*}
MODEL_FILENAME=$(basename -- "${MODEL_FILE}")
MODEL_EXTENSION="${MODEL_FILENAME##*.}"
MODEL_NAME="${MODEL_FILENAME%.*}"

OUTPUT_FILE_BASE="${MORPHEUS_ROOT}/.tmp/val_${MODEL_NAME}-"

# Load the utility scripts
source ${SCRIPT_DIR}/val-utils.sh
source ${SCRIPT_DIR}/val-run-pipeline.sh

if [[ "${RUN_PYTORCH}" = "1" ]]; then
   OUTPUT_FILE="${OUTPUT_FILE_BASE}pytorch.csv"

   run_pipeline_nlp_${SID_TYPE} \
      "${PCAP_INPUT_FILE}" \
      "inf-pytorch --model_filename=${MODEL_PYTORCH_FILE}" \
      "${OUTPUT_FILE}"

   # Get the diff
   PYTORCH_ERROR="${b}$(calc_error ${PCAP_TRUTH_FILE} ${OUTPUT_FILE})"
else
   PYTORCH_ERROR="${y}Skipped"
fi

if [[ "${RUN_TRITON_XGB}" = "1" ]]; then

   load_triton_model "abp-${ABP_TYPE}-xgb"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-xgb.csv"

   run_pipeline_abp_${ABP_TYPE} \
      "${INPUT_FILE}" \
      "inf-triton --model_name=abp-${ABP_TYPE}-xgb --server_url=localhost:8001 --force_convert_inputs=True" \
      "${OUTPUT_FILE}"

   # Get the diff
   TRITON_XGB_ERROR="${b}$(calc_error ${TRUTH_FILE} ${OUTPUT_FILE})"
else
   TRITON_XGB_ERROR="${y}Skipped"
fi

if [[ "${RUN_TRITON_TRT}" = "1" ]]; then
   load_triton_model "sid-${SID_TYPE}-trt"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-trt.csv"

   run_pipeline_nlp_${SID_TYPE} \
      "${PCAP_INPUT_FILE}" \
      "inf-triton --model_name=sid-${SID_TYPE}-trt --server_url=localhost:8001 --force_convert_inputs=True" \
      "${OUTPUT_FILE}"

   # Get the diff
   TRITON_TRT_ERROR="${b}$(calc_error ${PCAP_TRUTH_FILE} ${OUTPUT_FILE})"
else
   TRITON_TRT_ERROR="${y}Skipped"
fi

if [[ "${RUN_TENSORRT}" = "1" ]]; then
   # Generate the TensorRT model
   cd ${MORPHEUS_ROOT}/models/triton-model-repo/sid-${SID_TYPE}-trt/1

   echo "Generating the TensorRT model. This may take a minute..."
   morpheus tools onnx-to-trt --input_model ${MODEL_DIRECTORY}/${MODEL_NAME}.onnx --output_model ./sid-${SID_TYPE}-trt_b1-8_b1-16_b1-32.engine --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 256 --max_workspace_size 16000

   cd ${MORPHEUS_ROOT}

   load_triton_model "sid-${SID_TYPE}-trt"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}tensorrt.csv"

   run_pipeline_nlp_${SID_TYPE} \
      "${PCAP_INPUT_FILE}" \
      "inf-triton --model_name=sid-${SID_TYPE}-trt --server_url=localhost:8001 --force_convert_inputs=True" \
      "${OUTPUT_FILE}"

   # Get the diff
   TRITON_TRT_ERROR="${b}$(calc_error ${PCAP_TRUTH_FILE} ${OUTPUT_FILE})"

else
   TENSORRT_ERROR="${y}Skipped"
fi

echo -e "${b}===ERRORS===${x}"
echo -e "PyTorch     :${PYTORCH_ERROR}${x}"
echo -e "Triton(XGB) :${TRITON_XGB_ERROR}${x}"
echo -e "Triton(TRT) :${TRITON_TRT_ERROR}${x}"
echo -e "TensorRT    :${TENSORRT_ERROR}${x}"

echo -e "${g}Complete!${x}"
