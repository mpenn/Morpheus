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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace morpheus {
    /****** Component public implementations *******************/
    /****** CupyUtil****************************************/
    /**
     * @brief Structure that encapsulates cupy utilities.
     */
    struct CupyUtil {
        static pybind11::object cp_module; // handle to cupy module

        /**
         * TODO(Documentation)
         */
        static pybind11::module_ get_cp();

        /**
         * TODO(Documentation)
         */
        static pybind11::object tensor_to_cupy(const neo::TensorObject &tensor);

        /**
         * TODO(Documentation)
         */
        static neo::TensorObject cupy_to_tensor(pybind11::object cupy_array);
    };
}
