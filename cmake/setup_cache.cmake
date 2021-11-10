#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# ######################################################################################################################
# * CMake properties ------------------------------------------------------------------------------

list(APPEND CMAKE_MESSAGE_CONTEXT "cache")

set(NEO_CACHE_DIR
    "${CMAKE_SOURCE_DIR}/.cache"
    CACHE STRING "Directory to contain all CPM and CCache data")

function(configure_ccache)
  list(APPEND CMAKE_MESSAGE_CONTEXT "ccache")

  find_program(CCACHE_PROGRAM_PATH ccache)
  if(CCACHE_PROGRAM_PATH)
    message(STATUS "Using ccache: ${CCACHE_PROGRAM_PATH}")
    set(CCACHE_COMMAND CACHE STRING "${CCACHE_PROGRAM_PATH}")
    if(DEFINED ENV{CCACHE_DIR})
      message(
        STATUS
          "CCachEnvironment Variable 'CCACHE_DIR' is set. Using ccache directory: '$ENV{CCACHE_DIR}'. Ensure ccache.conf exists in directory."
      )
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM_PATH}")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM_PATH}")
      # set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM_PATH}) set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM_PATH})
    else()
      set(NEO_BASE_DIR "${CMAKE_SOURCE_DIR}")
      set(NEO_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/cmake")
      set(NEO_CMAKE_CCACHE_DIR "${NEO_CACHE_DIR}/ccache")

      message(STATUS "Using ccache directory: ${NEO_CMAKE_CCACHE_DIR}")
      # Write or update the ccache configuration file
      configure_file("${NEO_CMAKE_MODULES_PATH}/ccache.conf.in" "${NEO_CMAKE_CCACHE_DIR}/ccache.conf")
      set(ENV{CCACHE_CONFIGPATH} "${NEO_CMAKE_CCACHE_DIR}/ccache.conf")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK
                                   "CCACHE_CONFIGPATH=${NEO_CMAKE_CCACHE_DIR}/ccache.conf ${CCACHE_PROGRAM_PATH}")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE
                                   "CCACHE_CONFIGPATH=${NEO_CMAKE_CCACHE_DIR}/ccache.conf ${CCACHE_PROGRAM_PATH}")
      # set(CMAKE_C_COMPILER_LAUNCHER "CCACHE_CONFIGPATH=${NEO_CMAKE_CCACHE_DIR}/ccache.conf
      # ${CCACHE_PROGRAM_PATH}") set(CMAKE_CXX_COMPILER_LAUNCHER
      # "CCACHE_CONFIGPATH=${NEO_CMAKE_CCACHE_DIR}/ccache.conf ${CCACHE_PROGRAM_PATH}")
    endif(DEFINED ENV{CCACHE_DIR})
  else()
    message(WARNING "CCache option NEO_USE_CCACHE is enabled but ccache was not found. Check ccache installation.")
  endif(CCACHE_PROGRAM_PATH)

endfunction()

function(configure_cpm)
  list(APPEND CMAKE_MESSAGE_CONTEXT "cpm")

  # Set the CPM cache variable
  set(NEO_CPM_SOURCE_CACHE "${NEO_CACHE_DIR}/cpm")

  set(ENV{CPM_SOURCE_CACHE} ${NEO_CPM_SOURCE_CACHE})
  message(STATUS "Using CPM source cache: $ENV{CPM_SOURCE_CACHE}")

  # # Set the FetchContent default download folder to be the same as CPM
  # set(FETCHCONTENT_BASE_DIR "${NEO_CACHE_DIR}/fetch" CACHE STRING "" FORCE)

endfunction()

# Configure CCache if requested
if(NEO_USE_CCACHE)
  configure_ccache()
endif(NEO_USE_CCACHE)

configure_cpm()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
