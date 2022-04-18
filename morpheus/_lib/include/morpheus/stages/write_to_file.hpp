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

#include <morpheus/io/serializers.hpp>
#include <morpheus/messages/meta.hpp>
#include <morpheus/objects/file_types.hpp>
#include <morpheus/utilities/string_util.hpp>

#include <neo/core/segment.hpp>
#include <pyneo/node.hpp>

#include <fstream>
#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** WriteToFileStage********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class WriteToFileStage : public neo::pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = neo::pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    /**
     * TODO(Documentation)
     */
    WriteToFileStage(const neo::Segment &parent,
                     const std::string &name,
                     const std::string &filename,
                     std::ios::openmode mode = std::ios::out,
                     FileTypes file_type     = FileTypes::Auto);

  private:
    /**
     * TODO(Documentation)
     */
    void close();

    void write_json(reader_type_t &msg)
    {
        // Call df_to_json passing our fstream
        df_to_json(msg->get_info(), m_fstream);
    }

    void write_csv(reader_type_t &msg)
    {
        // Call df_to_csv passing our fstream
        df_to_csv(msg->get_info(), m_fstream, m_is_first);
    }

    operator_fn_t build_operator();

    bool m_is_first;
    std::ofstream m_fstream;
    std::function<void(reader_type_t &)> m_write_func;
};

/****** WriteToFileStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct WriteToFileStageInterfaceProxy
{
    /**
     * @brief Create and initialize a WriteToFileStage, and return the result.
     */
    static std::shared_ptr<WriteToFileStage> init(neo::Segment &parent,
                                                  const std::string &name,
                                                  const std::string &filename,
                                                  const std::string &mode = "w",
                                                  FileTypes file_type     = FileTypes::Auto);
};

#pragma GCC visibility pop
}  // namespace morpheus
