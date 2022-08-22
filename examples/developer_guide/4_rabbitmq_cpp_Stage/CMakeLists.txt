# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus_rabbit")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Add Morpheus Cmake helpers
set(MORPHEUS_ROOT $ENV{MORPHEUS_ROOT})
set(MORPHEUS_LIB_ROOT "${MORPHEUS_ROOT}/morpheus/_lib")
message(STATUS "MORPHEUS_ROOT=${MORPHEUS_ROOT}")
list(APPEND CMAKE_MODULE_PATH "${MORPHEUS_ROOT}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${MORPHEUS_ROOT}/cmake")

include(${MORPHEUS_ROOT}/morpheus/cmake/utils/python_module_tools.cmake)

# Project definition
project(rabbitmq_ex LANGUAGES C CXX CUDA)

# Configure all dependencies
include(cmake/dependencies.cmake)
set(Python3_FIND_VIRTUALENV "FIRST")
set(Python3_FIND_STRATEGY "LOCATION")
find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)
find_package(pybind11 REQUIRED)
find_package(rabbitmq REQUIRED)
find_package(SimpleAmqpClient REQUIRED)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# TODO: https://github.com/nv-morpheus/Morpheus/issues/331
link_directories(AFTER "${Python3_SITELIB}/morpheus/_lib")

pybind11_add_module(morpheus_rabbit MODULE "${CMAKE_CURRENT_SOURCE_DIR}/_lib/rabbitmq_source.cpp")

add_library(${PROJECT_NAME}::morpheus_rabbit ALIAS morpheus_rabbit)


target_link_libraries(morpheus_rabbit
    PUBLIC
      libcudacxx::libcudacxx
      morpheus
      rabbitmq
      SimpleAmqpClient
      srf::pysrf
)

target_include_directories(morpheus_rabbit
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${MORPHEUS_ROOT}/morpheus/_lib/include>
      rabbitmq
      ${SimpleAmqpClient_SOURCE_DIR}/src
)

set_target_properties(morpheus_rabbit PROPERTIES CXX_VISIBILITY_PRESET hidden)
inplace_build_copy(morpheus_rabbit "${CMAKE_CURRENT_SOURCE_DIR}/_lib")

message(STATUS " Install dest: (morpheus_rabbit) ${CMAKE_CURRENT_BINARY_DIR}")
install(
    TARGETS
    morpheus_rabbit
    LIBRARY DESTINATION
      "${CMAKE_CURRENT_BINARY_DIR}"
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
