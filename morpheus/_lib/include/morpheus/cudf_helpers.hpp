/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <glog/logging.h>
#include <pybind11/pytypes.h>
#include <cudf/io/types.hpp>

#include "./table_info.hpp"

// MUST COME LAST!!!
#include "cudf_helpers_api.h"

namespace morpheus {

void load_cudf_helpers()
{
    if (import_morpheus___lib__cudf_helpers() != 0)
    {
        pybind11::error_already_set ex;

        LOG(ERROR) << "Could not load cudf_helpers library: " << ex.what();
        throw ex;
    }
}
}  // namespace morpheus
