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

#include <neo/utils/string_utils.hpp>

#include <cstdint>
#include <filesystem>
#include <string>

namespace morpheus {

enum class FileTypes : int32_t
{
    Auto,
    JSON,
    CSV
};

FileTypes determine_file_type(std::string filename)
{
    auto filename_path = std::filesystem::path(filename);

    if (filename_path.extension() == ".json" || filename_path.extension() == ".jsonlines")
    {
        return FileTypes::JSON;
    }
    else if (filename_path.extension() == ".csv")
    {
        return FileTypes::CSV;
    }
    else
    {
        throw std::runtime_error(CONCAT_STR("Unsupported extension '"
                                            << filename_path.extension()
                                            << "' with 'auto' type. 'auto' only works with: csv, json"));
    }
}

}  // namespace morpheus
