#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e +o pipefail
# set -x
# set -v

function is_triton_running {
   res=$(curl -s -X GET \
        --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 2 \
        "http://localhost:8000/v2/health/ready" && echo "1" || echo "0")

   echo $res
}

function wait_for_triton {

   max_retries=${GPUCI_RETRY_MAX:=3}
   retries=0
   sleep_interval=${GPUCI_RETRY_SLEEP:=1}

   status=$(curl -X GET \
           -s -w "%{http_code}" \
           --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 2 \
           "http://localhost:8000/v2/health/ready" || echo 500)

   while [[ ${status} != 200 ]] && (( ${retries} < ${max_retries} )); do
      echo -e "${y}Triton not ready. Waiting ${sleep_interval} sec...${x}"

      sleep ${sleep_interval}

      status=$(curl -X GET \
              -s -w "%{http_code}" \
              --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 2 \
              "http://localhost:8000/v2/health/ready" || echo 500)

      retries=$((retries+1))
   done

   if [[ ${status} == 200 ]]; then
      echo -e "${g}Triton is ready.${x}"
      ret_val=0
   else
      echo -e "${g}Triton not ready.${x}"
      ret_val=1
   fi

   return ${ret_val}
}

function ensure_triton_running {

   TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:21.10-py3"}

   IS_RUNNING=$(is_triton_running)

   if [[ "${IS_RUNNING}" = "0" ]]; then

      # Kill the triton container on exit
      function cleanup {
         echo "Killing Triton container"
         docker kill triton-validation
      }

      trap cleanup EXIT

      echo "Launching Triton Container"

      # Launch triton container in explicit mode
      docker run --rm -ti -d --name=triton-validation --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v ${MORPHEUS_ROOT}/models:/models ${TRITON_IMAGE} tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit

      wait_for_triton
   fi
}

function load_triton_model {

   ensure_triton_running

   # Tell Triton to load the ONNX model
   curl --request POST \
      --url http://localhost:8000/v2/repository/models/${1}/load

   echo "Model '${1}' loaded in Triton"
}

function calc_error() {
   DIFF=$(diff $1 $2 | grep "^>" | wc -l)
   TOTAL=$(cat $1 | wc -l)
   ERROR=$(printf "%.2f\n" $(echo "${DIFF}/${TOTAL}*100.0" | bc -l))
   echo -e "${DIFF}/${TOTAL} (${ERROR} %)"
}

function calc_error_val() {
   DIFF=$(cat "${1}" | jq ".diff_rows")
   TOTAL=$(cat "${1}" | jq ".total_rows")
   ERROR=$(printf "%.2f\n" $(echo "${DIFF}/${TOTAL}*100.0" | bc -l))
   echo -e "${DIFF}/${TOTAL} (${ERROR} %)"
}