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

#include <neo/core/segment.hpp>
#include <pyneo/node.hpp>

#include <fstream>
#include <memory>
#include <regex>
#include <string>

namespace morpheus {
    /****** Component public implementations *******************/
    /****** SerializeStage********************************/
    /**
     * TODO(Documentation)
     */
#pragma GCC visibility push(default)
    class SerializeStage : public neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MessageMeta>> {
    public:
        using base_t = neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MessageMeta>>;
        using base_t::operator_fn_t;
        using base_t::reader_type_t;
        using base_t::writer_type_t;

        SerializeStage(const neo::Segment &parent,
                       const std::string &name,
                       const std::vector<std::string> &include,
                       const std::vector<std::string> &exclude,
                       bool fixed_columns = true);

    private:
        void make_regex_objs(const std::vector<std::string> &regex_strs, std::vector<std::regex> &regex_objs);

        bool match_column(const std::vector<std::regex> &patterns, const std::string &column) const;

        bool include_column(const std::string &column) const;

        bool exclude_column(const std::string &column) const;

        TableInfo get_meta(reader_type_t &msg);

        operator_fn_t build_operator();

        bool m_fixed_columns;
        std::vector<std::regex> m_include;
        std::vector<std::regex> m_exclude;
        std::vector<std::string> m_column_names;
    };

    /****** WriteToFileStageInterfaceProxy******************/
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct SerializeStageInterfaceProxy {
        /**
         * @brief Create and initialize a SerializeStage, and return the result.
         */
        static std::shared_ptr<SerializeStage> init(neo::Segment &parent,
                                                    const std::string &name,
                                                    const std::vector<std::string> &include,
                                                    const std::vector<std::string> &exclude,
                                                    bool fixed_columns = true);
    };

#pragma GCC visibility pop
}  // namespace morpheus
