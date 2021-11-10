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

function(find_and_configure_cudf version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "cudf")

  include(get_cpm)

  # Need to set CUB_DIR since cuDF doesnt work well with CUB/Thrust dependencies
  rapids_cpm_find(cudf ${version}
    GLOBAL_TARGETS
      cudf cudf::cudf
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/rapidsai/cudf
      GIT_TAG         branch-${CUDF_VERSION}
      GIT_SHALLOW     TRUE
      SOURCE_SUBDIR   cpp
      PATCH_COMMAND   git apply --whitespace=fix ${PROJECT_SOURCE_DIR}/cmake/deps/patches/cudf.patch
      OPTIONS         "CUDF_ENABLE_ARROW_S3 OFF"
                      "PER_THREAD_DEFAULT_STREAM ON"
                      "BUILD_TESTS OFF"
                      "CUB_DIR ${CUDAToolkit_INCLUDE_DIRS}/cub/cmake"
  )

endfunction()

find_and_configure_cudf(${CUDF_VERSION})
