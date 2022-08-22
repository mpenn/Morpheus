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

#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/file_types.hpp"

#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <fstream>
#include <functional>  // for function
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** WriteToFileStage********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class WriteToFileStage : public srf::pysrf::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * TODO(Documentation)
     */
    WriteToFileStage(const std::string &filename,
                     std::ios::openmode mode = std::ios::out,
                     FileTypes file_type     = FileTypes::Auto,
                     bool include_index_col  = true);

  private:
    /**
     * TODO(Documentation)
     */
    void close();

    void write_json(sink_type_t &msg);

    void write_csv(sink_type_t &msg);

    subscribe_fn_t build_operator();

    bool m_is_first;
    bool m_include_index_col;
    std::ofstream m_fstream;
    std::function<void(sink_type_t &)> m_write_func;
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
    static std::shared_ptr<srf::segment::Object<WriteToFileStage>> init(srf::segment::Builder &builder,
                                                                        const std::string &name,
                                                                        const std::string &filename,
                                                                        const std::string &mode = "w",
                                                                        FileTypes file_type     = FileTypes::Auto,
                                                                        bool include_index_col  = true);
};

#pragma GCC visibility pop
}  // namespace morpheus
