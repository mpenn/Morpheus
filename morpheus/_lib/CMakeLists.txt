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

# Cudf_helpers needs to come first
morpheus_add_cython_libraries(./cudf_helpers.pyx
  LINK_TARGETS cudf::cudf Python3::NumPy
  OUTPUT_TARGET cudf_helpers_target
)

morpheus_add_pybind11_module(./common.cpp
  LINK_TARGETS ${cudf_helpers_target}
)

morpheus_add_pybind11_module(./messages.cpp LINK_TARGETS ${cudf_helpers_target})
morpheus_add_pybind11_module(./stages.cpp LINK_TARGETS ${cudf_helpers_target} TritonClient::httpclient)
