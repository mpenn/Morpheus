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

#pragma once

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** FiberQueue****************************************/
/**
 * TODO(Documentation)
 */
class FiberQueue
{
  public:
    FiberQueue(std::size_t max_size);

    /**
     * TODO(Documentation)
     */
    boost::fibers::channel_op_status put(pybind11::object &&item, bool block = true, float timeout = 0.0);

    /**
     * TODO(Documentation)
     */
    boost::fibers::channel_op_status get(pybind11::object &item, bool block = true, float timeout = 0.0);

    /**
     * TODO(Documentation)
     */
    void close();

    /**
     * TODO(Documentation)
     */
    bool is_closed();

    /**
     * TODO(Documentation)
     */
    void join();

  private:
    boost::fibers::buffered_channel<pybind11::object> m_queue;
};

#pragma GCC visibility push(default)
/****** FiberQueueInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct FiberQueueInterfaceProxy
{
    /**
     * @brief Create and initialize a FIberQueue, and return a shared pointer to the result.
     */
    static std::shared_ptr<morpheus::FiberQueue> init(std::size_t max_size);

    /**
     * TODO(Documentation)
     */
    static void put(morpheus::FiberQueue &self, pybind11::object item, bool block = true, float timeout = 0.0);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get(morpheus::FiberQueue &self, bool block = true, float timeout = 0.0);

    /**
     * TODO(Documentation)
     */
    static void close(morpheus::FiberQueue &self);
};
#pragma GCC visibility pop
}  // namespace morpheus
