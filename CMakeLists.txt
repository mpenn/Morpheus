# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
