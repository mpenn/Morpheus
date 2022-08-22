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

list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus_example")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add Morpheus Cmake helpers
set(MORPHEUS_ROOT $ENV{MORPHEUS_ROOT})
set(MORPHEUS_LIB_ROOT "${MORPHEUS_ROOT}/morpheus/_lib")
message(STATUS "MORPHEUS_ROOT=${MORPHEUS_ROOT}")
list(APPEND CMAKE_MODULE_PATH "${MORPHEUS_ROOT}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${MORPHEUS_ROOT}/cmake")

# Add the RAPIDS cmake helper scripts
include(import-rapids-cmake)

# Configure CUDA architecture
include(configure_cuda_architecture)

include(get_cpm)
rapids_cpm_init(OVERRIDE "${MORPHEUS_ROOT}/cmake/deps/rapids_cpm_package_overrides.json")

find_package(CUDAToolkit REQUIRED)

include(${MORPHEUS_ROOT}/morpheus/cmake/utils/python_module_tools.cmake)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Project definition
project(morpheus_example LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED) # Required by Morpheus. Fail early if we don't have it.
set(LIBCUDACXX_VERSION "1.6.0" CACHE STRING "Version of libcudacxx to use")
include(deps/Configure_libcudacxx)

set(SRF_VERSION 22.06 CACHE STRING "Which version of SRF to use")
include(deps/Configure_srf)

find_package(pybind11 REQUIRED)

message(STATUS "*******************************")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "MORPHEUS_LIB_ROOT=${MORPHEUS_LIB_ROOT}")
morpheus_add_pybind11_module(
    morpheus_example
    MODULE_ROOT
      "${CMAKE_CURRENT_SOURCE_DIR}/_lib"
    SOURCE_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/_lib/pass_thru.cpp"
    INCLUDE_DIRS
      "${MORPHEUS_LIB_ROOT}/include"
    LINK_TARGETS
      morpheus
      srf::pysrf
    OUTPUT_TARGET
      morpheus_example_target
    INSTALL_DEST
      ${CMAKE_CURRENT_BINARY_DIR}
)

#pybind11_add_module(morpheus_example MODULE "${CMAKE_CURRENT_SOURCE_DIR}/_lib/pass_thru.cpp")

#add_library(${PROJECT_NAME}::morpheus_example ALIAS morpheus_example)


#target_link_libraries(morpheus_example
#    PUBLIC
#      libcudacxx::libcudacxx
#      morpheus
#      srf::pysrf
#)

#target_include_directories(morpheus_example
#    PUBLIC
#      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#      $<BUILD_INTERFACE:${MORPHEUS_ROOT}/morpheus/_lib/include>
#)


#set_target_properties(morpheus_example PROPERTIES CXX_VISIBILITY_PRESET hidden)
#inplace_build_copy(${morpheus_example_target} "${CMAKE_CURRENT_SOURCE_DIR}/_lib")

message(STATUS " Install dest: (morpheus_example) ${CMAKE_CURRENT_BINARY_DIR}")
#install(
#    TARGETS
#    morpheus_example
#    LIBRARY DESTINATION
#      "${CMAKE_CURRENT_BINARY_DIR}"
#)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
