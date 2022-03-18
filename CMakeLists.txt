# SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus")

# Global options
option(BUILD_SHARED_LIBS "Default value for whether or not to build shared or static libraries" ON)
option(MORPHEUS_BUILD_BENCHMARKS "Whether or not to build benchmarks" OFF)
option(MORPHEUS_BUILD_EXAMPLES "Whether or not to build examples" OFF)
option(MORPHEUS_BUILD_TESTS "Whether or not to build tests" OFF)
option(MORPHEUS_USE_CCACHE "Enable caching compilation results with ccache" OFF)
option(MORPHEUS_USE_CLANG_TIDY "Enable running clang-tidy as part of the build process" ON)
option(MORPHEUS_USE_CONDA "Enables finding dependencies via conda instead of vcpkg.
  Note: This will disable vcpkg. All dependencies must be installed first in the conda environment" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

enable_testing()

# CMake path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Add the RAPIDS cmake helper scripts
include(import-rapids-cmake)

# Configure CUDA architecture
include(configure_cuda_architecture)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Toolchain selection (Conda/VCPKG/custom) and configuration
include(cmake/configure_toolchain.cmake)

# Project definition
project(morpheus LANGUAGES C CXX CUDA)

# Ccache configuration
include(setup_cache)

# Configure all dependencies
include(cmake/dependencies.cmake)

# After dependencies have been created, set the clang-tidy option only on our code
if(MORPHEUS_USE_CLANG_TIDY)
  # Disabled global clang-tidy for now. Once trtlab has been moved to neo, can be re-enabled
  # set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
  message(STATUS "Enabling clang-tidy for targets that opt-in")
endif()

# Include the main trtlab code
add_subdirectory(morpheus)

# Optionally build examples
if (NEO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

##################################
##### Morpheus Python Setup ######
##################################
set(MORPHEUS_PYTHON_FILES "")

# Manually install the below files. install(DIRECTORY) doesnt work well and
# makes it impossible to get these files and neo/*.py in one command.
list(PREPEND MORPHEUS_PYTHON_FILES
  "pyproject.toml"
  "setup.cfg"
  "setup.py"
  "versioneer.py"
)

install(
  FILES ${MORPHEUS_PYTHON_FILES}
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# Ensure that all python files get installed somewhere. TBD on whether
# ${CMAKE_CURRENT_BINARY_DIR} is the correct place
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/morpheus
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILES_MATCHING
    PATTERN "*.py"
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
