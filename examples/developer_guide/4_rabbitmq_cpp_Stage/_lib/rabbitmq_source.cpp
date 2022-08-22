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

#include "rabbitmq_source.hpp"

#include <neo/core/segment.hpp>
#include <neo/core/segment_object.hpp>

#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>

#include <glog/logging.h>
#include <pybind11/chrono.h>  // for timedelta->chrono
#include <pybind11/pybind11.h>
#include <boost/fiber/operations.hpp>  // for this_fiber::sleep_for

#include <exception>
#include <sstream>
#include <vector>

namespace morpheus_rabbit {

RabbitMQSourceStage::RabbitMQSourceStage(const neo::Segment &segment,
                                         const std::string &name,
                                         const std::string &host,
                                         const std::string &exchange,
                                         const std::string &exchange_type,
                                         const std::string &queue_name,
                                         std::chrono::milliseconds poll_interval) :
  neo::SegmentObject(segment, name),
  base_t(segment, name, build_observable()),
  m_channel{AmqpClient::Channel::Create(host)},
  m_poll_interval{poll_interval}
{
    m_channel->DeclareExchange(exchange, exchange_type);
    m_queue_name = m_channel->DeclareQueue(queue_name);
    m_channel->BindQueue(m_queue_name, exchange);
}

neo::Observable<RabbitMQSourceStage::source_type_t> RabbitMQSourceStage::build_observable()
{
    return neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t> &subscriber) {
        try
        {
            this->source_generator(subscriber);
        } catch (const std::exception &e)
        {
            LOG(ERROR) << "Encountered error while polling RabbitMQ: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            close();
            return;
        }

        close();
        subscriber.on_completed();
    });
}

void RabbitMQSourceStage::source_generator(neo::Subscriber<RabbitMQSourceStage::source_type_t> &subscriber)
{
    const std::string consumer_tag = m_channel->BasicConsume(m_queue_name, "", true, false);
    while (subscriber.is_subscribed())
    {
        AmqpClient::Envelope::ptr_t envelope;
        if (m_channel->BasicConsumeMessage(consumer_tag, envelope, 0))
        {
            try
            {
                auto table   = from_json(envelope->Message()->Body());
                auto message = MessageMeta::create_from_cpp(std::move(table), 0);
                subscriber.on_next(std::move(message));
            } catch (const std::exception &e)
            {
                LOG(ERROR) << "Error occurred converting RabbitMQ message to Dataframe: " << e.what();
            }
            m_channel->BasicAck(envelope);
        }
        else
        {
            // Sleep when there are no messages
            boost::this_fiber::sleep_for(m_poll_interval);
        }
    }
}

cudf::io::table_with_metadata RabbitMQSourceStage::from_json(const std::string &body) const
{
    cudf::io::source_info source{body.c_str(), body.size()};
    auto options = cudf::io::json_reader_options::builder(source).lines(true);
    return cudf::io::read_json(options.build());
}

void RabbitMQSourceStage::close()
{
    // disconnect
    if (m_channel)
    {
        m_channel.reset();
    }
}

// ************ WriteToFileStageInterfaceProxy ************* //
std::shared_ptr<RabbitMQSourceStage> RabbitMQSourceStageInterfaceProxy::init(neo::Segment &segment,
                                                                             const std::string &name,
                                                                             const std::string &host,
                                                                             const std::string &exchange,
                                                                             const std::string &exchange_type,
                                                                             const std::string &queue_name,
                                                                             std::chrono::milliseconds poll_interval)
{
    auto stage =
        std::make_shared<RabbitMQSourceStage>(segment, name, host, exchange, exchange_type, queue_name, poll_interval);
    segment.register_node<RabbitMQSourceStage>(stage);
    return stage;
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(morpheus_rabbit, m)
{
    py::class_<RabbitMQSourceStage, neo::SegmentObject, std::shared_ptr<RabbitMQSourceStage>>(
        m, "RabbitMQSourceStage", py::multiple_inheritance())
        .def(py::init<>(&RabbitMQSourceStageInterfaceProxy::init),
             py::arg("segment"),
             py::arg("name"),
             py::arg("host"),
             py::arg("exchange"),
             py::arg("exchange_type") = "fanout",
             py::arg("queue_name")    = "",
             py::arg("poll_interval") = 100ms);
}

}  // namespace morpheus_rabbit
