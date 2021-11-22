#!/bin/bash

set -e +o pipefail
# set -x
# set -v


# RUN OPTIONS
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Call minibert first
${SCRIPT_DIR}/val-sid.sh "minibert"
${SCRIPT_DIR}/val-sid.sh "bert"
