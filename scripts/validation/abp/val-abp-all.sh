#!/bin/bash

set -e +o pipefail
# set -x
# set -v

# RUN OPTIONS
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Now call the script to run
${SCRIPT_DIR}/val-abp.sh "nvsmi"
