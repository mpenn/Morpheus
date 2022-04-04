
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

#include <neo/core/tensor.hpp>
#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>

#include <pybind11/pytypes.h>


#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace morpheus {
    /****** Component public implementations *******************/
    /****** neo::TensorObject****************************************/
    // TODO(Devin) defined in neo
    /****** <NAME>InterfaceProxy *************************/
#pragma GCC visibility push(default)
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct NeoTensorObjectInterfaceProxy {
        static pybind11::dict cuda_array_interface(neo::TensorObject &self);
    };
#pragma GCC visibility pop
}
