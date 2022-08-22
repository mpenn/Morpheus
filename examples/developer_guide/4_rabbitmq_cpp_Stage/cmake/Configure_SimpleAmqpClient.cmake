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

function(find_and_configure_SimpleAmqpClient version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "SimpleAmqpClient")

  find_package(rabbitmq REQUIRED)

  rapids_cpm_find(SimpleAmqpClient ${version}
    GLOBAL_TARGETS
    SimpleAmqpClient SimpleAmqpClient::SimpleAmqpClient
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/alanxz/SimpleAmqpClient
      GIT_TAG         "v${version}"
      GIT_SHALLOW     TRUE
      OPTIONS         "Rabbitmqc_INCLUDE_DIR ${rabbitmq_SOURCE_DIR}/librabbitmq"
                      "Rabbitmqc_LIBRARY ${rabbitmq_BINARY_DIR}/librabbitmq/librabbitmq.so"
                      "BUILD_API_DOCS OFF"
  )

  if(SimpleAmqpClient_ADDED)
    set(SimpleAmqpClient_SOURCE_DIR "${SimpleAmqpClient_SOURCE_DIR}" PARENT_SCOPE)
    set(SimpleAmqpClient_BINARY_DIR "${SimpleAmqpClient_BINARY_DIR}" PARENT_SCOPE)
  endif()

endfunction()

find_and_configure_SimpleAmqpClient(${SIMPLE_AMQP_CLIENT_VERSION})
