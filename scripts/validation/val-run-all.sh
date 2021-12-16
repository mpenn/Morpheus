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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure the .tmp folder exists
if [[ ! -d "${SCRIPT_DIR}/../../.tmp" ]]; then
   mkdir -p "${SCRIPT_DIR}/../../.tmp"
fi

# Run everything once USE_CPP=False
export USE_CPP=0

${SCRIPT_DIR}/abp/val-abp-all.sh
${SCRIPT_DIR}/hammah/val-hammah-all.sh
${SCRIPT_DIR}/phishing/val-phishing-all.sh
${SCRIPT_DIR}/sid/val-sid-all.sh

# Run everything once USE_CPP=True
export USE_CPP=1

${SCRIPT_DIR}/abp/val-abp-all.sh
${SCRIPT_DIR}/hammah/val-hammah-all.sh
${SCRIPT_DIR}/phishing/val-phishing-all.sh
${SCRIPT_DIR}/sid/val-sid-all.sh
