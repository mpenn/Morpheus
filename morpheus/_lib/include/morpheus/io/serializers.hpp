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

#pragma once

#include "morpheus/objects/table_info.hpp"

#include <ostream>
#include <string>

namespace morpheus {

std::string df_to_csv(const TableInfo& tbl, bool include_header, bool include_index_col = true);

void df_to_csv(const TableInfo& tbl, std::ostream& out_stream, bool include_header, bool include_index_col = true);

// Note the include_index_col is currently being ignored in both versions of `df_to_json` due to a known issue in
// Pandas: https://github.com/pandas-dev/pandas/issues/37600
std::string df_to_json(const TableInfo& tbl, bool include_index_col = true);

void df_to_json(const TableInfo& tbl, std::ostream& out_stream, bool include_index_col = true);

}  // namespace morpheus
