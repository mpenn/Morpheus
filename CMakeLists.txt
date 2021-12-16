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

# Use Vcpkg if variable is set. Must be done before first call to project()!
# This will automatically install all dependencies in vcpkg.json
if(DEFINED CMAKE_TOOLCHAIN_FILE)
  message(STATUS "Using custom toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
elseif (DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Skipping Vcpkg toolchain. Conda environment detected which takes precidence.")
elseif(DEFINED ENV{VCPKG_ROOT})
  if(NOT EXISTS "$ENV{VCPKG_ROOT}")
    message(FATAL_ERROR "Vcpkg env 'VCPKG_ROOT' set to '$ENV{VCPKG_ROOT}' but file does not exist! Exiting...")
    return()
  endif()
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  set(USING_VCPKG True)
  message(STATUS "Vcpkg detected, CMAKE_TOOLCHAIN_FILE set to: ${CMAKE_TOOLCHAIN_FILE}")
endif()
enable_testing()

# Help vcpkg out on CI systems by ensuring the cache directory exists
if(DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE} AND NOT EXISTS "$ENV{VCPKG_DEFAULT_BINARY_CACHE}")
  message(STATUS "Vcpkg binary cache missing. Creating directory. Cache location: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
  file(MAKE_DIRECTORY "$ENV{VCPKG_DEFAULT_BINARY_CACHE}")
endif()

# CMake path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Add the RAPIDS cmake helper scripts
include(import-rapids-cmake)

# Default to using "" for CUDA_ARCHITECTURES to build based on GPU in system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES was not defined. Defaulting to '' to build only for local architecture. Specify -DCMAKE_CUDA_ARCHITECTURES='ALL' to build for all archs.")
endif()

rapids_cuda_init_architectures(morpheus)

if(USING_VCPKG)
  message(STATUS "Creating project. If this hangs, check 'VCPKG_ROOT' environment variable. Should not take more than a few seconds to see additional output")
endif()

# Global options
option(BUILD_SHARED_LIBS "Default value for whether or not to build shared or static libraries" ON)
option(MORPHEUS_BUILD_BENCHMARKS "Whether or not to build benchmarks" OFF)
option(MORPHEUS_BUILD_EXAMPLES "Whether or not to build examples" OFF)
option(MORPHEUS_BUILD_TESTS "Whether or not to build tests" OFF)
option(MORPHEUS_USE_CCACHE "Enable caching compilation results with ccache" OFF)
option(MORPHEUS_USE_CLANG_TIDY "Enable running clang-tidy as part of the build process" ON)

# If using shared libs (the default) use a custom triplet file to use dynamic linking
if(BUILD_SHARED_LIBS)
  set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg_triplets")
  set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic")
endif()

project(morpheus LANGUAGES C CXX CUDA)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Once the build type is set, remove any dumb vcpkg debug folders from the
# search paths. Without this FindBoost fails since it defaults to the debug
# binaries
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
  message(STATUS "Release Build: Removing debug paths from CMAKE_PREFIX_PATH and CMAKE_FIND_ROOT_PATH")
  list(REMOVE_ITEM CMAKE_PREFIX_PATH "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug")
  list(REMOVE_ITEM CMAKE_FIND_ROOT_PATH "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(setup_cache)

# Configure conda
if(DEFINED ENV{CONDA_PREFIX})
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE STRING "" FORCE)
  endif()

  message(STATUS "Conda environment detected, prepending CONDA_PREFIX to CMAKE_FIND_ROOT_PATH")
  list(PREPEND CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX}")
  list(REMOVE_DUPLICATES CMAKE_FIND_ROOT_PATH)
endif()

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
