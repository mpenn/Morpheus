# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "dep")

# Needed force rapids cpm to use our source directory.
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cache/cpm")

# Add the RAPIDS cmake helper scripts
include(import-rapids-cmake)
rapids_cpm_init()

# Configure CUDA architecture
include(configure_cuda_architecture)

include(get_cpm)
rapids_cpm_init(OVERRIDE "${MORPHEUS_ROOT}/cmake/deps/rapids_cpm_package_overrides.json")

find_package(CUDAToolkit REQUIRED) # Required by Morpheus. Fail early if we don't have it.
set(LIBCUDACXX_VERSION "1.6.0" CACHE STRING "Version of libcudacxx to use")
include(deps/Configure_libcudacxx)

set(SRF_VERSION 22.06 CACHE STRING "Which version of SRF to use")
include(deps/Configure_srf)

set(RABBITMQ_VERSION "0.11.0" CACHE STRING "Version of RabbitMQ-C to use")
include(Configure_rabbitmq)

set(SIMPLE_AMQP_CLIENT_VERSION "2.5.1" CACHE STRING "Version of SimpleAmqpClient to use")
include(Configure_SimpleAmqpClient)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
