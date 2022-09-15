#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_rabbitmq version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "rabbitmq")

  rapids_cpm_find(rabbitmq ${version}
    GLOBAL_TARGETS
      rabbitmq rabbitmq::rabbitmq
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/alanxz/rabbitmq-c
      GIT_TAG         "v${version}"
      GIT_SHALLOW     TRUE
      OPTIONS         "BUILD_EXAMPLES OFF"
                      "BUILD_TESTING OFF"
                      "BUILD_TOOLS OFF"
  )

endfunction()

find_and_configure_rabbitmq(${RABBITMQ_VERSION})
