# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# Create the utils library first
add_library(utils SHARED
    src/matx_functions.cu
    src/serializers.cpp
    src/type_utils.cu
)

target_include_directories(utils
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(utils
    PUBLIC
      neo::pyneo
      matx::matx
      cudf::cudf
      Python3::NumPy
      pybind11::pybind11
)

set_target_properties(utils
    PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_utils
)

# Cudf_helpers needs to come first
morpheus_add_cython_libraries(./cudf_helpers.pyx
    LINK_TARGETS utils
    OUTPUT_TARGET cudf_helpers_target
)

morpheus_add_pybind11_module(./common.cpp
    LINK_TARGETS
      ${cudf_helpers_target}
      utils
)

morpheus_add_pybind11_module(./file_types.cpp
)

morpheus_add_pybind11_module(./messages.cpp
    LINK_TARGETS
      ${cudf_helpers_target}
      utils
)

morpheus_add_pybind11_module(./stages.cpp
    LINK_TARGETS
      ${cudf_helpers_target}
      TritonClient::httpclient
      utils
      RDKAFKA::RDKAFKA
)

install(
    TARGETS utils pyneo
    EXPORT ${PROJECT_NAME}-targets
)

install(
    TARGETS pyneo
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
