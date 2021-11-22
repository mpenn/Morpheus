#!/bin/bash

set -e +o pipefail
# set -x
# set -v

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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
