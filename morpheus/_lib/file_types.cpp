/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <morpheus/file_types.hpp>

#include <neo/utils/string_utils.hpp>

#include <pybind11/pybind11.h>

#include <filesystem>
#include <stdexcept>

namespace morpheus {

namespace py = pybind11;

PYBIND11_MODULE(file_types, m)
{
    py::enum_<FileTypes>(m,
                         "FileTypes",
                         "The type of files that the `FileSourceStage` can read and `WriteToFileStage` can write. Use "
                         "'auto' to determine from the file extension.")
        .value("Auto", FileTypes::Auto)
        .value("JSON", FileTypes::JSON)
        .value("CSV", FileTypes::CSV);

    m.def("determine_file_type", &determine_file_type);

}  // module
}  // namespace morpheus
