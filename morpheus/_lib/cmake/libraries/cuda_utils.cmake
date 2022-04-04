# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

add_library(cuda_utils
    SHARED
      ${MORPHEUS_LIB_ROOT}/src/objects/dev_mem_info.cpp
      ${MORPHEUS_LIB_ROOT}/src/objects/table_info.cpp
      ${MORPHEUS_LIB_ROOT}/src/utilities/matx_util.cu
      ${MORPHEUS_LIB_ROOT}/src/utilities/type_util.cu
)

target_include_directories(cuda_utils
    PUBLIC
      "${MORPHEUS_LIB_ROOT}/include"
)

target_link_libraries(cuda_utils
    PUBLIC
      neo::pyneo
      matx::matx
      cudf::cudf
      Python3::NumPy
)

set_target_properties(cuda_utils
    PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_utils
)

message(STATUS " Install dest: (cuda_utils) ${CMAKE_CURRENT_BINARY_DIR}")
install(
    TARGETS
      cuda_utils
    LIBRARY DESTINATION
      "${CMAKE_CURRENT_BINARY_DIR}"
)

if (MORPHEUS_PYTHON_INPLACE_BUILD)
  inplace_build_copy(cuda_utils ${MORPHEUS_LIB_ROOT})
endif()
