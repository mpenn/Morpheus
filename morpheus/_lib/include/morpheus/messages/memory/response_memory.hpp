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

#include <morpheus/objects/tensor.hpp>

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

#include <string>
#include <vector>

namespace morpheus {
    /****** Component public implementations *******************/
    /****** ResponseMemory****************************************/
    /**
     * TODO(Documentation)
     */
    class ResponseMemory {
    public:
        ResponseMemory(size_t count);

        size_t count{0};
        std::map<std::string, neo::TensorObject> outputs;

        /**
         * TODO(Documentation)
         */
        bool has_output(const std::string &name) const;
    };

    /****** ResponseMemoryInterfaceProxy *************************/
#pragma GCC visibility push(default)
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct ResponseMemoryInterfaceProxy {
        static pybind11::object get_output(ResponseMemory &self, const std::string &name);

        static neo::TensorObject get_output_tensor(ResponseMemory &self, const std::string &name);
    };
#pragma GCC visibility pop
}