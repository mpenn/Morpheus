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

#include <morpheus/messages/multi.hpp>
#include <morpheus/messages/multi_inference.hpp>

#include <pyneo/node.hpp>

#include <string>
#include <memory>


namespace morpheus {
    /****** Component public implementations *******************/
    /****** PreprocessFILStage**********************************/
    /**
     * TODO(Documentation)
     */
#pragma GCC visibility push(default)
    class PreprocessFILStage
            : public neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>> {
    public:
        using base_t = neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
        using base_t::operator_fn_t;
        using base_t::reader_type_t;
        using base_t::writer_type_t;

        PreprocessFILStage(const neo::Segment &parent, const std::string &name);

    private:
        /**
         * TODO(Documentation)
         */
        operator_fn_t build_operator();

        std::string m_vocab_file;
    };

    /****** PreprocessFILStageInferenceProxy********************/
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct PreprocessFILStageInterfaceProxy {
        /**
         * @brief Create and initialize a PreprocessFILStage, and return the result.
         */
        static std::shared_ptr<PreprocessFILStage> init(neo::Segment &parent, const std::string &name);
    };
#pragma GCC visibility pop
}