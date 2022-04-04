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


#include <morpheus/messages/meta.hpp>
#include <morpheus/messages/multi.hpp>

#include <neo/core/segment.hpp>
#include <pyneo/node.hpp>

#include <string>
#include <memory>


namespace morpheus {
    /****** Component public implementations *******************/
    /****** DeserializationStage********************************/
    /**
     * TODO(Documentation)
     */
#pragma GCC visibility push(default)
    class DeserializeStage
            : public neo::pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>> {
    public:
        using base_t = neo::pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>>;
        using base_t::operator_fn_t;
        using base_t::reader_type_t;
        using base_t::writer_type_t;

        DeserializeStage(const neo::Segment &parent, const std::string &name, size_t batch_size);

    private:
        /**
         * TODO(Documentation)
         */
        operator_fn_t build_operator();

        size_t m_batch_size;
    };

    /****** DeserializationStageInterfaceProxy******************/
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct DeserializeStageInterfaceProxy {
        /**
         * @brief Create and initialize a DeserializationStage, and return the result.
         */
        static std::shared_ptr<DeserializeStage> init(neo::Segment &parent, const std::string &name, size_t batch_size);
    };
#pragma GCC visibility pop
}