/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>  // IWYU pragma: keep

#include <string>

#pragma once

namespace morpheus {
/****** Component public implementations *******************/
/****** CuDFTableUtil****************************************/
/**
 * @brief Structure that encapsulates cuDF table utilities.
 */
struct CuDFTableUtil
{
    /**
     * TODO(Documentation)
     */
    static cudf::io::table_with_metadata load_table(const std::string &filename);
};
}  // namespace morpheus
