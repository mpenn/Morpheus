/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <SimpleAmqpClient/SimpleAmqpClient.h>  // for AmqpClient::Channel::ptr_t
#include <cudf/io/types.hpp>                    // for cudf::io::table_with_metadata
#include <morpheus/messages/meta.hpp>           // for MessageMeta
#include <pysrf/node.hpp>                       // for srf::pysrf::PythonSource
#include <srf/segment/builder.hpp>              // for Segment Builder
#include <srf/segment/object.hpp>               // for Segment Object

#include <chrono>  // for chrono::milliseconds
#include <memory>  // for shared_ptr
#include <string>

namespace morpheus_rabbit {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace std::literals;
using namespace morpheus;

class RabbitMQSourceStage : public srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    RabbitMQSourceStage(const std::string &host,
                        const std::string &exchange,
                        const std::string &exchange_type        = "fanout"s,
                        const std::string &queue_name           = ""s,
                        std::chrono::milliseconds poll_interval = 100ms);

    ~RabbitMQSourceStage() override = default;

  private:
    subscriber_fn_t build();
    void source_generator(rxcpp::subscriber<source_type_t> subscriber);
    cudf::io::table_with_metadata from_json(const std::string &body) const;
    void close();

    std::chrono::milliseconds m_poll_interval;
    std::string m_queue_name;
    AmqpClient::Channel::ptr_t m_channel;
};

/****** RabbitMQSourceStageInferenceProxy**********************/
/**
 * @brief Interface proxy, used to insulate Python bindings.
 */
struct RabbitMQSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a RabbitMQSourceStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<RabbitMQSourceStage>> init(srf::segment::Builder &builder,
                                                                           const std::string &name,
                                                                           const std::string &host,
                                                                           const std::string &exchange,
                                                                           const std::string &exchange_type,
                                                                           const std::string &queue_name,
                                                                           std::chrono::milliseconds poll_interval);
};
#pragma GCC visibility pop
}  // namespace morpheus_rabbit
