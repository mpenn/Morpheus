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

#include <morpheus/common.hpp>
#include <morpheus/messages.hpp>
#include <morpheus/type_utils.hpp>

#include <pyneo/node.hpp>
#include <neo/core/segment_object.hpp>
#include <neo/forward.hpp>
#include <neo/utils/type_utils.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <nvtext/subword_tokenize.hpp>

#include <glog/logging.h>
#include <http_client.h>
#include <librdkafka/rdkafkacpp.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <thrust/iterator/constant_iterator.h>
#include <boost/fiber/recursive_mutex.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <utility>



namespace morpheus {

using namespace pybind11::literals;

template <typename FuncT, typename SeqT>
auto foreach_map(const SeqT& seq, FuncT func)
{
    using value_t  = typename SeqT::const_reference;
    using return_t = decltype(func(std::declval<value_t>()));

    std::vector<return_t> result{};

    std::transform(seq.cbegin(), seq.cend(), std::back_inserter(result), func);

    return result;
}

template <typename FuncT, typename SeqT>
auto foreach_map2(const SeqT& seq, FuncT func)
{
    using value_t  = typename SeqT::const_reference;
    using return_t = decltype(func(std::declval<value_t>()));

    std::vector<return_t> result{};

    std::transform(seq.begin(), seq.end(), std::back_inserter(result), func);

    return result;
}

class FileSourceStage : public neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>;
    using base_t::source_type_t;

    FileSourceStage(const neo::Segment& parent, const std::string& name, std::string filename, int repeat = 1) :
      neo::SegmentObject(parent, name),
      base_t(parent, name),
      m_filename(std::move(filename)),
      m_repeat(repeat)
    {
        this->set_source_observable(neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t>& sub) {
            auto data_table = this->load_table();

            // Using 0 will default to creating a new range index
            int index_col_count = 0;

            // Check if we have a first column with INT64 data type
            if (data_table.metadata.column_names.size() >= 1 &&
                data_table.tbl->get_column(0).type().id() == cudf::type_id::INT64)
            {
                std::regex index_regex(R"((unnamed: 0|id))",
                                       std::regex_constants::ECMAScript | std::regex_constants::icase);

                // Get the column name
                auto col_name = data_table.metadata.column_names[0];

                // Check it against some common terms
                if (std::regex_search(col_name, index_regex))
                {
                    // Also, if its the hideous 'Unnamed: 0', then just use an empty string
                    if (col_name == "Unnamed: 0")
                    {
                        data_table.metadata.column_names[0] = "";
                    }

                    index_col_count = 1;
                }
            }

            // Next, create the message metadata. This gets reused for repeats
            auto meta = MessageMeta::create_from_cpp(std::move(data_table), index_col_count);

            // Always push at least 1
            sub.on_next(meta);

            for (cudf::size_type repeat_idx = 1; repeat_idx < m_repeat; ++repeat_idx)
            {
                // Clone the previous meta object
                {
                    pybind11::gil_scoped_acquire gil;

                    // Use the copy function
                    auto df = meta->get_py_table().attr("copy")();

                    pybind11::int_ df_len = pybind11::len(df);

                    pybind11::object index = df.attr("index");

                    df.attr("index") = index + df_len;

                    meta = MessageMeta::create_from_python(std::move(df));
                }

                sub.on_next(meta);
            }

            sub.on_completed();
        }));
    }

  private:
    cudf::io::table_with_metadata load_table()
    {
        auto file_path = std::filesystem::path(m_filename);

        if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines")
        {
            // First, load the file into json
            auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{m_filename}).lines(true);

            auto tbl = cudf::io::read_json(options.build());

            auto found = std::find(tbl.metadata.column_names.begin(), tbl.metadata.column_names.end(), "data");

            if (found == tbl.metadata.column_names.end())
                return tbl;

            // Super ugly but cudf cant handle newlines and add extra escapes. So we need to convert
            // \\n -> \n
            // \\/ -> \/
            auto columns = tbl.tbl->release();

            size_t idx = found - tbl.metadata.column_names.begin();

            auto updated_data = cudf::strings::replace(
                cudf::strings_column_view{columns[idx]->view()}, cudf::string_scalar("\\n"), cudf::string_scalar("\n"));

            updated_data = cudf::strings::replace(
                cudf::strings_column_view{updated_data->view()}, cudf::string_scalar("\\/"), cudf::string_scalar("/"));

            columns[idx] = std::move(updated_data);

            tbl.tbl = std::move(std::make_unique<cudf::table>(std::move(columns)));

            return tbl;
        }
        else if (file_path.extension() == ".csv")
        {
            auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{m_filename});

            return cudf::io::read_csv(options.build());
        }
        else
        {
            LOG(FATAL) << "Unknown extension for file: " << m_filename;
            throw std::runtime_error("Unknown extension");
        }
    }

    std::string m_filename;
    int m_repeat{1};
};

#define CHECK_KAFKA(command, expected, msg)                                                                    \
    {                                                                                                          \
        RdKafka::ErrorCode __code = command;                                                                   \
        if (__code != expected)                                                                                \
        {                                                                                                      \
            LOG(ERROR) << msg << ". Received unexpected ErrorCode. Expected: " << #expected << "(" << expected \
                       << "), Received: " << __code << ", Msg: " << RdKafka::err2str(__code);                  \
        }                                                                                                      \
    }

class KafkaSourceStage : public neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>;
    using base_t::source_type_t;

    KafkaSourceStage(const neo::Segment& parent,
                     const std::string& name,
                     size_t max_batch_size,
                     std::string topic,
                     int32_t batch_timeout_ms,
                     std::map<std::string, std::string> config,
                     bool disable_commit = false) :
      neo::SegmentObject(parent, name),
      base_t(parent, name),
      m_max_batch_size(max_batch_size),
      m_topic(std::move(topic)),
      m_batch_timeout_ms(batch_timeout_ms),
      m_config(std::move(config)),
      m_disable_commit(disable_commit)
    {
        this->set_source_observable(neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t>& sub) {
            // Build rebalancer
            Rebalancer rebalancer(*this, [&sub, this](std::vector<std::unique_ptr<RdKafka::Message>>& message_batch) {
                // If we are unsubscribed, throw an error to break the loops
                if (!sub.is_subscribed())
                {
                    throw UnsubscribedException();
                }

                if (!message_batch.empty())
                {
                    auto batch = this->process_batch(std::move(message_batch));

                    sub.on_next(std::move(batch));

                    if (m_requires_commit)
                    {
                        return true;
                    }
                }

                return false;
            });

            // Build consumer
            auto consumer = this->create_consumer(&rebalancer);

            rebalancer.rebalance_loop(consumer.get());

            consumer->unsubscribe();

            consumer->close();

            consumer.reset();

            sub.on_completed();
        }));
    }

    ~KafkaSourceStage() override = default;

  protected:
  private:
    class UnsubscribedException : public std::exception
    {};

    void start() override
    {
        // Save off the queues before setting our concurrency back to 1
        for (size_t i = 0; i < this->concurrency(); ++i)
        {
            m_task_queues.push_back(this->resources().fiber_pool().next_task_queue());
        }

        this->concurrency(1);

        // Call the default start
        neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>::start();
    }

    class Rebalancer : public RdKafka::RebalanceCb
    {
      public:
        Rebalancer(KafkaSourceStage& parent,
                   std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>>&)> process_fn) :
          m_parent(parent),
          m_process_fn(std::move(process_fn))
        {}

        void rebalance_cb(RdKafka::KafkaConsumer* consumer,
                          RdKafka::ErrorCode err,
                          std::vector<RdKafka::TopicPartition*>& partitions) override
        {
            std::unique_lock<boost::fibers::recursive_mutex> lock(m_mutex);

            if (err == RdKafka::ERR__ASSIGN_PARTITIONS)
            {
                VLOG(10) << m_parent.display_str("Rebalance: Assign Partitions");

                // application may load offets from arbitrary external storage here and update \p partitions
                if (consumer->rebalance_protocol() == "COOPERATIVE")
                {
                    CHECK_KAFKA(std::unique_ptr<RdKafka::Error>(consumer->incremental_assign(partitions))->code(),
                                RdKafka::ERR_NO_ERROR,
                                "Error during incremental assign");
                }
                else
                {
                    CHECK_KAFKA(consumer->assign(partitions), RdKafka::ERR_NO_ERROR, "Error during assign");
                }

                std::vector<std::function<bool()>> tasks;

                for (auto partition : partitions)
                {
                    auto queue_ptr = consumer->get_partition_queue(partition);

                    // Now forward to one of the running queues
                    // queue->forward(m_parent.m_queues[i % m_parent.m_queues.size()].get());
                    queue_ptr->forward(nullptr);

                    auto partition_ptr = RdKafka::TopicPartition::create(
                        partition->topic(), partition->partition(), partition->offset());

                    tasks.emplace_back([q = queue_ptr, p = partition_ptr, consumer, this]() {
                        auto partition = std::unique_ptr<RdKafka::TopicPartition>(p);
                        auto queue     = std::unique_ptr<RdKafka::Queue>(q);

                        while (m_is_rebalanced)
                        {
                            // Build the batch
                            auto messages = this->partition_progress_step(queue.get());

                            try
                            {
                                // Process the messages. Returns true if we need to commit
                                auto should_commit = m_process_fn(messages);

                                if (should_commit)
                                {
                                    int64_t max_offset = -1000;
                                    for (auto& m : messages)
                                    {
                                        DCHECK(m->partition() == partition->partition())
                                            << "Inconsistent error. Message partition does not match fiber partition";

                                        max_offset = std::max(max_offset, m->offset());
                                    }

                                    // Find the last message for this partition
                                    partition->set_offset(max_offset + 1);

                                    CHECK_KAFKA(
                                        consumer->commitAsync(std::vector<RdKafka::TopicPartition*>{partition.get()}),
                                        RdKafka::ERR_NO_ERROR,
                                        "Error during commitAsync");
                                }
                            } catch (UnsubscribedException&)
                            {
                                // Return false for unsubscribed error
                                return false;
                            }
                        }

                        // Return true if we exited normally
                        return true;
                    });
                }

                // Set this before launching the tasks
                m_is_rebalanced = true;

                m_partition_future = std::move(m_parent.launch_tasks(std::move(tasks)));
            }
            else if (err == RdKafka::ERR__REVOKE_PARTITIONS)
            {
                VLOG(10) << m_parent.display_str("Rebalance: Revoke Partitions");

                // Application may commit offsets manually here if auto.commit.enable=false
                if (consumer->rebalance_protocol() == "COOPERATIVE")
                {
                    CHECK_KAFKA(std::unique_ptr<RdKafka::Error>(consumer->incremental_unassign(partitions))->code(),
                                RdKafka::ERR_NO_ERROR,
                                "Error during incremental unassign");
                }
                else
                {
                    CHECK_KAFKA(consumer->unassign(), RdKafka::ERR_NO_ERROR, "Error during unassign");
                }

                // Stop all processing queues
                m_is_rebalanced = false;

                // Wait until all processing has completed
                if (m_partition_future.valid())
                {
                    m_partition_future.wait();
                }
            }
            else
            {
                LOG(ERROR) << "Rebalancing error: " << RdKafka::err2str(err) << std::endl;
                CHECK_KAFKA(consumer->unassign(), RdKafka::ERR_NO_ERROR, "Error during unassign");
            }
        }

        void rebalance_loop(RdKafka::KafkaConsumer* consumer)
        {
            do
            {
                // Poll until we are rebalanced
                while (!this->is_rebalanced())
                {
                    VLOG(10) << m_parent.display_str("Rebalance: Calling poll to trigger rebalance");
                    consumer->poll(500);
                }
            } while (m_partition_future.get());
        }

        bool is_rebalanced()
        {
            std::unique_lock<boost::fibers::recursive_mutex> lock(m_mutex);

            return m_is_rebalanced;
        }

      private:
        std::vector<std::unique_ptr<RdKafka::Message>> partition_progress_step(RdKafka::Queue* queue)
        {
            auto batch_timeout = std::chrono::milliseconds(m_parent.m_batch_timeout_ms);

            size_t msg_count = 0;
            std::vector<std::unique_ptr<RdKafka::Message>> messages;

            auto now       = std::chrono::high_resolution_clock::now();
            auto batch_end = now + batch_timeout;

            do
            {
                auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - now).count();

                DCHECK(remaining_ms >= 0) << "Cant have negative reminaing time";

                std::unique_ptr<RdKafka::Message> msg{queue->consume(std::min(10L, remaining_ms))};

                switch (msg->err())
                {
                case RdKafka::ERR__TIMED_OUT:
                    // Yield on a timeout
                    boost::this_fiber::yield();
                    break;
                case RdKafka::ERR_NO_ERROR:

                    // VLOG(10) << this->display_str(
                    //     CONCAT_STR("Got message. Topic: " << msg->topic_name() << ", Part: " <<
                    //     msg->partition()
                    //                                       << ", Offset: " << msg->offset()));

                    messages.emplace_back(std::move(msg));
                    break;
                case RdKafka::ERR__PARTITION_EOF:
                    VLOG_EVERY_N(10, 10) << "Hit EOF for partition";
                    // Hit the end, sleep for 100 ms
                    boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
                    break;
                default:
                    /* Errors */
                    LOG(ERROR) << "Consume failed: " << msg->errstr();
                }

                // Update now
                now = std::chrono::high_resolution_clock::now();
            } while (msg_count < m_parent.m_max_batch_size && now < batch_end && m_is_rebalanced);

            return std::move(messages);
        }

        KafkaSourceStage& m_parent;
        std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>>&)> m_process_fn;
        boost::fibers::recursive_mutex m_mutex;
        bool m_is_rebalanced{false};
        neo::SharedFuture<bool> m_partition_future;
    };

    std::unique_ptr<RdKafka::Conf> build_kafka_conf(const std::map<std::string, std::string>& config_in)
    {
        // Copy the config
        std::map<std::string, std::string> config_out(config_in);

        std::map<std::string, std::string> defaults{{"session.timeout.ms", "60000"},
                                                    {"enable.auto.commit", "false"},
                                                    {"auto.offset.reset", "latest"},
                                                    {"enable.partition.eof", "true"}};

        // Set some defaults if they dont exist
        config_out.merge(defaults);

        m_requires_commit = config_out["enable.auto.commit"] == "false";

        if (m_requires_commit && m_disable_commit)
        {
            LOG(WARNING) << "KafkaSourceStage: Commits have been disabled for this Kafka consumer. This should only be "
                            "used in a debug environment";
            m_requires_commit = false;
        }
        else if (!m_requires_commit && m_disable_commit)
        {
            // User has auto-commit on and disable commit at same time
            LOG(WARNING) << "KafkaSourceStage: The config option 'enable.auto.commit' was set to True but commits have "
                            "been disabled for this Kafka consumer. This should only be used in a debug environment";
        }

        // Make the kafka_conf and set all properties
        auto kafka_conf = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

        for (auto const& key_value : config_out)
        {
            std::string error_string;
            if (RdKafka::Conf::ConfResult::CONF_OK != kafka_conf->set(key_value.first, key_value.second, error_string))
            {
                LOG(ERROR) << "Error occurred while setting Kafka configuration. Error: " << error_string;
            }
        }

        return std::move(kafka_conf);
    }

    neo::SharedFuture<bool> launch_tasks(std::vector<std::function<bool()>>&& tasks)
    {
        std::vector<neo::SharedFuture<bool>> partition_futures;

        // Loop over tasks enqueuing onto saved fiber queues
        for (size_t i = 0; i < tasks.size(); ++i)
        {
            partition_futures.emplace_back(
                this->m_task_queues[i % this->m_task_queues.size()]->enqueue(std::move(tasks[i])));
        }

        // Launch a new task to await on the futures
        return this->m_task_queues[tasks.size() % this->m_task_queues.size()]->enqueue([partition_futures]() {
            bool ret_val = true;

            for (auto& f : partition_futures)
            {
                if (!f.get())
                {
                    // Return false if any return false
                    ret_val = false;
                }
            }

            return ret_val;
        });
    }

    std::unique_ptr<RdKafka::KafkaConsumer> create_consumer(Rebalancer* rebalancer)
    {
        auto kafka_conf = this->build_kafka_conf(m_config);
        std::string errstr;

        if (RdKafka::Conf::ConfResult::CONF_OK != kafka_conf->set("rebalance_cb", rebalancer, errstr))
        {
            LOG(FATAL) << "Error occurred while setting Kafka rebalance function. Error: " << errstr;
        }

        auto consumer =
            std::unique_ptr<RdKafka::KafkaConsumer>(RdKafka::KafkaConsumer::create(kafka_conf.get(), errstr));

        if (!consumer)
        {
            LOG(FATAL) << "Error occurred creating Kafka consumer. Error: " << errstr;
        }

        // Subscribe to the topic. Uses the default rebalancer
        CHECK_KAFKA(consumer->subscribe(std::vector<std::string>{m_topic}),
                    RdKafka::ERR_NO_ERROR,
                    "Error subscribing to topics");

        auto spec_topic =
            std::unique_ptr<RdKafka::Topic>(RdKafka::Topic::create(consumer.get(), m_topic, nullptr, errstr));

        RdKafka::Metadata* md;
        CHECK_KAFKA(consumer->metadata(spec_topic == nullptr, spec_topic.get(), &md, 1000),
                    RdKafka::ERR_NO_ERROR,
                    "Failed to list_topics in Kafka broker");

        std::map<std::string, std::vector<int32_t>> topic_parts;

        VLOG(10) << this->display_str(CONCAT_STR("Subscribed to " << md->topics()->size() << " topics:"));

        for (auto const& topic : *(md->topics()))
        {
            auto& part_ids = topic_parts[topic->topic()];

            auto const& parts = *(topic->partitions());

            std::transform(parts.cbegin(), parts.cend(), std::back_inserter(part_ids), [](auto const& part) {
                return part->id();
            });

            auto toppar_list = foreach_map(parts, [&topic](const auto& part) {
                return std::unique_ptr<RdKafka::TopicPartition>{
                    RdKafka::TopicPartition::create(topic->topic(), part->id())};
            });

            std::vector<RdKafka::TopicPartition*> toppar_ptrs =
                foreach_map(toppar_list, [](const std::unique_ptr<RdKafka::TopicPartition>& x) { return x.get(); });

            // Query Kafka to populate the TopicPartitions with the desired offsets
            CHECK_KAFKA(consumer->committed(toppar_ptrs, 1000),
                        RdKafka::ERR_NO_ERROR,
                        "Failed retrieve Kafka committed offsets");

            auto committed =
                foreach_map(toppar_list, [](const std::unique_ptr<RdKafka::TopicPartition>& x) { return x->offset(); });

            // Query Kafka to populate the TopicPartitions with the desired offsets
            CHECK_KAFKA(consumer->position(toppar_ptrs), RdKafka::ERR_NO_ERROR, "Failed retrieve Kafka positions");

            auto positions =
                foreach_map(toppar_list, [](const std::unique_ptr<RdKafka::TopicPartition>& x) { return x->offset(); });

            VLOG(10) << this->display_str(CONCAT_STR(
                "   Topic: '" << topic->topic() << "', Parts: " << array_to_str(part_ids.begin(), part_ids.end())
                              << ", Committed: " << array_to_str(committed.begin(), committed.end())
                              << ", Positions: " << array_to_str(positions.begin(), positions.end())));
        }

        return std::move(consumer);
    }

    cudf::io::table_with_metadata load_table(const std::string& buffer)
    {
        auto options =
            cudf::io::json_reader_options::builder(cudf::io::source_info(buffer.c_str(), buffer.size())).lines(true);

        auto tbl = cudf::io::read_json(options.build());

        auto found = std::find(tbl.metadata.column_names.begin(), tbl.metadata.column_names.end(), "data");

        if (found == tbl.metadata.column_names.end())
            return tbl;

        // Super ugly but cudf cant handle newlines and add extra escapes. So we need to convert
        // \\n -> \n
        // \\/ -> \/
        auto columns = tbl.tbl->release();

        size_t idx = found - tbl.metadata.column_names.begin();

        auto updated_data = cudf::strings::replace(
            cudf::strings_column_view{columns[idx]->view()}, cudf::string_scalar("\\n"), cudf::string_scalar("\n"));

        updated_data = cudf::strings::replace(
            cudf::strings_column_view{updated_data->view()}, cudf::string_scalar("\\/"), cudf::string_scalar("/"));

        columns[idx] = std::move(updated_data);

        tbl.tbl = std::move(std::make_unique<cudf::table>(std::move(columns)));

        return tbl;
    }

    std::shared_ptr<morpheus::MessageMeta> process_batch(std::vector<std::unique_ptr<RdKafka::Message>>&& message_batch)
    {
        std::ostringstream buffer;

        // Build the string for the batch
        for (auto& msg : message_batch)
        {
            buffer << static_cast<char*>(msg->payload()) << "\n";
        }

        auto data_table = this->load_table(buffer.str());

        // Next, create the message metadata. This gets reused for repeats
        auto meta = MessageMeta::create_from_cpp(std::move(data_table), 0);

        return meta;
    }

    size_t m_max_batch_size{128};
    std::string m_topic{"test_pcap"};
    uint32_t m_batch_timeout_ms{100};
    std::map<std::string, std::string> m_config;
    bool m_disable_commit{false};

    bool m_requires_commit{false};  // Whether or not manual committing is required
    std::vector<std::shared_ptr<neo::TaskQueue<neo::FiberMetaData>>> m_task_queues;

    friend Rebalancer;
};

class DeserializeStage : public neo::pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>>
{
  public:
    using base_t = neo::pyneo::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    DeserializeStage(const neo::Segment& parent, const std::string& name, size_t batch_size) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_batch_size(batch_size)
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    // Make one large MultiMessage
                    auto full_message = std::make_shared<MultiMessage>(x, 0, x->count());

                    // Loop over the MessageMeta and create sub-batches
                    for (size_t i = 0; i < x->count(); i += this->m_batch_size)
                    {
                        auto next = full_message->get_slice(i, std::min(i + this->m_batch_size, x->count()));

                        output.on_next(std::move(next));
                    }
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    size_t m_batch_size;
};

class PreprocessNLPStage
  : public neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    PreprocessNLPStage(const neo::Segment& parent,
                       const std::string& name,
                       std::string vocab_hash_file,
                       uint32_t sequence_length,
                       bool truncation,
                       bool do_lower_case,
                       bool add_special_token,
                       int stride = -1) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_vocab_hash_file(std::move(vocab_hash_file)),
      m_sequence_length(sequence_length),
      m_truncation(truncation),
      m_do_lower_case(do_lower_case),
      m_add_special_token(add_special_token),
      m_stride(stride)
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            uint32_t stride = m_stride;

            // Auto calc stride to be 75% of sequence length
            if (stride < 0)
            {
                stride = m_sequence_length / 2;
                stride = stride + stride / 2;
            }

            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, stride, &output](reader_type_t&& x) {
                    // Convert to string view
                    auto string_col = cudf::strings_column_view{x->get_meta("data").get_column(0)};

                    // Create the hashed vocab
                    thread_local std::unique_ptr<nvtext::hashed_vocabulary> vocab =
                        nvtext::load_vocabulary_file(this->m_vocab_hash_file);

                    // Perform the tokenizer
                    auto token_results = nvtext::subword_tokenize(string_col,
                                                                  *vocab,
                                                                  this->m_sequence_length,
                                                                  stride,
                                                                  this->m_do_lower_case,
                                                                  this->m_truncation,
                                                                  string_col.size() * 2);

                    // Build the results
                    auto memory = std::make_shared<InferenceMemory>(token_results.nrows_tensor);

                    int32_t length = token_results.tensor_token_ids->size() / token_results.sequence_length;
                    auto input_ids_released =
                        cudf::cast(token_results.tensor_token_ids->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();

                    memory->inputs["input_ids"] = std::move(Tensor::create(
                        std::move(input_ids_released.data),
                        DType::create<int32_t>(),
                        std::vector<neo::TensorIndex>{length, static_cast<int>(token_results.sequence_length)},
                        std::vector<neo::TensorIndex>{},
                        0));

                    length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
                    auto input_mask_released =
                        cudf::cast(token_results.tensor_attention_mask->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->inputs["input_mask"] = std::move(Tensor::create(
                        std::move(input_mask_released.data),
                        DType::create<int32_t>(),
                        std::vector<neo::TensorIndex>{length, static_cast<int>(token_results.sequence_length)},
                        std::vector<neo::TensorIndex>{},
                        0));

                    length = token_results.tensor_metadata->size() / 3;
                    auto seq_ids_released =
                        cudf::cast(token_results.tensor_metadata->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->inputs["seq_ids"] =
                        std::move(Tensor::create(std::move(seq_ids_released.data),
                                                 DType::create<int32_t>(),
                                                 std::vector<neo::TensorIndex>{length, static_cast<int32_t>(3)},
                                                 std::vector<neo::TensorIndex>{},
                                                 0));

                    auto next = std::make_shared<MultiInferenceMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                    output.on_next(std::move(next));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_vocab_hash_file;
    uint32_t m_sequence_length;
    bool m_truncation;
    bool m_do_lower_case;
    bool m_add_special_token;
    int m_stride{-1};
};

class PreprocessFILStage
  : public neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    PreprocessFILStage(const neo::Segment& parent, const std::string& name) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator())
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    std::vector<std::string> fea_cols = {
                        "nvidia_smi_log.gpu.pci.tx_util",
                        "nvidia_smi_log.gpu.pci.rx_util",
                        "nvidia_smi_log.gpu.fb_memory_usage.used",
                        "nvidia_smi_log.gpu.fb_memory_usage.free",
                        "nvidia_smi_log.gpu.bar1_memory_usage.total",
                        "nvidia_smi_log.gpu.bar1_memory_usage.used",
                        "nvidia_smi_log.gpu.bar1_memory_usage.free",
                        "nvidia_smi_log.gpu.utilization.gpu_util",
                        "nvidia_smi_log.gpu.utilization.memory_util",
                        "nvidia_smi_log.gpu.temperature.gpu_temp",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold",
                        "nvidia_smi_log.gpu.temperature.memory_temp",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold",
                        "nvidia_smi_log.gpu.power_readings.power_draw",
                        "nvidia_smi_log.gpu.clocks.graphics_clock",
                        "nvidia_smi_log.gpu.clocks.sm_clock",
                        "nvidia_smi_log.gpu.clocks.mem_clock",
                        "nvidia_smi_log.gpu.clocks.video_clock",
                        "nvidia_smi_log.gpu.applications_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.applications_clocks.mem_clock",
                        "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.default_applications_clocks.mem_clock",
                        "nvidia_smi_log.gpu.max_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.max_clocks.sm_clock",
                        "nvidia_smi_log.gpu.max_clocks.mem_clock",
                        "nvidia_smi_log.gpu.max_clocks.video_clock",
                        "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock",
                    };
                    // TODO(MDD): Add some sort of lock here to prevent fixing columns after they have been accessed
                    auto df_meta           = x->get_meta(fea_cols);
                    auto df_meta_col_names = df_meta.get_column_names();

                    auto packed_data = std::make_shared<rmm::device_buffer>(
                        fea_cols.size() * x->mess_count * sizeof(float), rmm::cuda_stream_per_thread);

                    std::vector<std::string> bad_cols;

                    auto df_just_features = df_meta.get_view();

                    for (size_t i = 0; i < df_meta.num_columns(); ++i)
                    {
                        if (df_just_features.column(df_meta.num_indices() + i).type().id() == cudf::type_id::STRING)
                        {
                            bad_cols.push_back(df_meta_col_names[i]);
                        }
                    }

                    // Need to ensure all string columns have been converted to numbers. This requires running a
                    // regex which is too difficult to do from C++ at this time. So grab the GIL, make the
                    // conversions, and release. This is horribly inefficient, but so is the JSON lines format for
                    // this workflow
                    if (!bad_cols.empty())
                    {
                        pybind11::gil_scoped_acquire gil;

                        pybind11::object df = x->meta->get_py_table();

                        std::string regex = R"((\d+))";

                        for (auto c : bad_cols)
                        {
                            df[pybind11::str(c)] = df[pybind11::str(c)]
                                                 .attr("str")
                                                 .attr("extract")(pybind11::str(regex), "expand"_a = true)
                                                 .attr("astype")(pybind11::str("float32"));
                        }

                        // Now re-get the meta
                        df_meta          = x->get_meta(fea_cols);
                        df_just_features = df_meta.get_view();
                    }

                    for (size_t i = 0; i < df_meta.num_columns(); ++i)
                    {
                        auto curr_col = df_just_features.column(df_meta.num_indices() + i);

                        auto curr_ptr = static_cast<float*>(packed_data->data()) + i * df_just_features.num_rows();

                        // Check if we are something other than float
                        if (curr_col.type().id() != cudf::type_id::FLOAT32)
                        {
                            auto float_data = cudf::cast(curr_col, cudf::data_type(cudf::type_id::FLOAT32))->release();

                            // Do the copy here before it goes out of scipe
                            NEO_CHECK_CUDA(cudaMemcpy(curr_ptr,
                                                      float_data.data->data(),
                                                      df_just_features.num_rows() * sizeof(float),
                                                      cudaMemcpyDeviceToDevice));
                        }
                        else
                        {
                            NEO_CHECK_CUDA(cudaMemcpy(curr_ptr,
                                                      curr_col.data<float>(),
                                                      df_just_features.num_rows() * sizeof(float),
                                                      cudaMemcpyDeviceToDevice));
                        }
                    }

                    // Need to do a transpose here
                    auto transposed_data =
                        transpose(DevMemInfo{x->mess_count * fea_cols.size(), neo::TypeId::FLOAT32, packed_data, 0},
                                  fea_cols.size(),
                                  x->mess_count);

                    auto input__0 = Tensor::create(transposed_data,
                                                   DType::create<float>(),
                                                   std::vector<neo::TensorIndex>{static_cast<long long>(x->mess_count),
                                                                                 static_cast<int>(fea_cols.size())},
                                                   std::vector<neo::TensorIndex>{},
                                                   0);

                    auto seg_ids = Tensor::create(
                        create_seg_ids(x->mess_count, fea_cols.size(), neo::TypeId::UINT32),
                        DType::create<uint32_t>(),
                        std::vector<neo::TensorIndex>{static_cast<long long>(x->mess_count), static_cast<int>(3)},
                        std::vector<neo::TensorIndex>{},
                        0);

                    // Build the results
                    auto memory = std::make_shared<InferenceMemoryFIL>(x->mess_count, input__0, seg_ids);

                    auto next = std::make_shared<MultiInferenceMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                    output.on_next(std::move(next));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_vocab_file;
};

void __checkTritonErrors(triton::client::Error status,
                         const std::string& methodName,
                         const std::string& filename,
                         const int& lineNumber)
{
    if (!status.IsOk())
    {
        std::string err_msg =
            CONCAT_STR("Triton Error while executing '" << methodName << "'. Error: " + status.Message() << "\n"
                                                        << filename << "(" << lineNumber << ")");
        LOG(ERROR) << err_msg;
        throw std::runtime_error(err_msg);
    }
}

#define CHECK_TRITON(method) __checkTritonErrors(method, #method, __FILE__, __LINE__);

class InferenceClientStage
  : public neo::pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>
{
  public:
    using base_t = neo::pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    InferenceClientStage(const neo::Segment& parent,
                         const std::string& name,
                         std::string model_name,
                         std::string server_url,
                         bool force_convert_inputs,
                         bool use_shared_memory,
                         bool needs_logits,
                         std::map<std::string, std::string> inout_mapping = {}) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_model_name(std::move(model_name)),
      m_server_url(std::move(server_url)),
      m_force_convert_inputs(force_convert_inputs),
      m_use_shared_memory(use_shared_memory),
      m_needs_logits(needs_logits),
      m_inout_mapping(std::move(inout_mapping)),
      m_options(m_model_name)
    {
        // Connect with the server to setup the inputs/outputs
        this->connect_with_server();
    }

  private:
    struct TritonInOut
    {
        std::string name;
        size_t bytes;
        DType datatype;
        std::vector<int> shape;
        std::string mapped_name;
        size_t offset;
    };

    bool is_default_grpc_port(std::string& server_url)
    {
        // Check if we are the default gRPC port of 8001 and try 8000 for http client instead
        size_t colon_loc = server_url.find_last_of(':');

        if (colon_loc == -1)
        {
            return false;
        }

        // Check if the port matches 8001
        if (server_url.size() < colon_loc + 1 || server_url.substr(colon_loc + 1) != "8001")
        {
            return false;
        }

        // It matches, change to 8000
        server_url = server_url.substr(0, colon_loc) + ":8000";

        return true;
    }

    void connect_with_server()
    {
        std::string server_url = m_server_url;

        std::unique_ptr<triton::client::InferenceServerHttpClient> client;

        auto result = triton::client::InferenceServerHttpClient::Create(&client, server_url, false);

        // Now load the input/outputs for the model
        bool is_server_live = false;

        triton::client::Error status = client->IsServerLive(&is_server_live);

        if (!status.IsOk())
        {
            if (this->is_default_grpc_port(server_url))
            {
                LOG(WARNING) << "Failed to connect to Triton at '" << m_server_url
                             << "'. Default gRPC port of (8001) was detected but C++ "
                                "InferenceClientStage uses HTTP protocol. Retrying with default HTTP port (8000)";

                // We are using the default gRPC port, try the default HTTP
                std::unique_ptr<triton::client::InferenceServerHttpClient> unique_client;

                auto result = triton::client::InferenceServerHttpClient::Create(&unique_client, server_url, false);

                client = std::move(unique_client);

                status = client->IsServerLive(&is_server_live);
            }
            else if (status.Message().find("Unsupported protocol") != std::string::npos)
            {
                throw std::runtime_error(
                    CONCAT_STR("Failed to connect to Triton at '"
                               << m_server_url
                               << "'. Received 'Unsupported Protocol' error. Are you using the right port? The C++ "
                                  "InferenceClientStage uses Triton's HTTP protocol instead of gRPC. Ensure you have "
                                  "specified the HTTP port (Default 8000)."));
            }

            if (!status.IsOk())
                throw std::runtime_error(CONCAT_STR("Unable to connect to Triton at '"
                                                    << m_server_url
                                                    << "'. Check the URL and port and ensure the server is running."));
        }

        // Save this for new clients
        m_server_url = server_url;

        if (!is_server_live)
            throw std::runtime_error("Server is not live");

        bool is_server_ready = false;
        CHECK_TRITON(client->IsServerReady(&is_server_ready));

        if (!is_server_ready)
            throw std::runtime_error("Server is not ready");

        bool is_model_ready = false;
        CHECK_TRITON(client->IsModelReady(&is_model_ready, this->m_model_name));

        if (!is_model_ready)
            throw std::runtime_error("Model is not ready");

        std::string model_metadata_json;
        CHECK_TRITON(client->ModelMetadata(&model_metadata_json, this->m_model_name));

        auto model_metadata = nlohmann::json::parse(model_metadata_json);

        std::string model_config_json;
        CHECK_TRITON(client->ModelConfig(&model_config_json, this->m_model_name));

        auto model_config = nlohmann::json::parse(model_config_json);

        if (model_config.contains("max_batch_size"))
        {
            m_max_batch_size = model_config.at("max_batch_size").get<int>();
        }

        for (auto const& input : model_metadata.at("inputs"))
        {
            auto shape = input.at("shape").get<std::vector<int>>();

            auto dtype = DType::from_triton(input.at("datatype").get<std::string>());

            size_t bytes = dtype.item_size();

            for (auto& y : shape)
            {
                if (y == -1)
                {
                    y = m_max_batch_size;
                }

                bytes *= y;
            }

            std::string mapped_name = input.at("name").get<std::string>();

            if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
            {
                mapped_name = m_inout_mapping[mapped_name];
            }

            m_model_inputs.push_back(TritonInOut{input.at("name").get<std::string>(),
                                                 bytes,
                                                 DType::from_triton(input.at("datatype").get<std::string>()),
                                                 shape,
                                                 mapped_name,
                                                 0});
        }

        for (auto const& output : model_metadata.at("outputs"))
        {
            auto shape = output.at("shape").get<std::vector<int>>();

            auto dtype = DType::from_triton(output.at("datatype").get<std::string>());

            size_t bytes = dtype.item_size();

            for (auto& y : shape)
            {
                if (y == -1)
                {
                    y = m_max_batch_size;
                }

                bytes *= y;
            }

            std::string mapped_name = output.at("name").get<std::string>();

            if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
            {
                mapped_name = m_inout_mapping[mapped_name];
            }

            m_model_outputs.push_back(
                TritonInOut{output.at("name").get<std::string>(), bytes, dtype, shape, mapped_name, 0});
        }
    }

    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            std::unique_ptr<triton::client::InferenceServerHttpClient> client;

            CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, m_server_url, false));

            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output, &client](reader_type_t&& x) {
                    auto reponse_memory = std::make_shared<ResponseMemory>(x->count);

                    // Create the output memory blocks
                    for (auto& model_output : m_model_outputs)
                    {
                        auto total_shape = model_output.shape;

                        // First dimension will always end up being the number of rows
                        total_shape[0] = x->count;

                        auto elem_count =
                            std::accumulate(total_shape.begin(), total_shape.end(), 1, std::multiplies<>());

                        // Create the output memory
                        auto output_buffer = std::make_shared<rmm::device_buffer>(
                            elem_count * model_output.datatype.item_size(), rmm::cuda_stream_per_thread);

                        reponse_memory->outputs[model_output.mapped_name] =
                            Tensor::create(std::move(output_buffer),
                                           model_output.datatype,
                                           std::vector<neo::TensorIndex>{static_cast<int>(total_shape[0]),
                                                                         static_cast<int>(total_shape[1])},
                                           std::vector<neo::TensorIndex>{},
                                           0);
                    }

                    // This will be the final output of all mini-batches
                    auto response = std::make_shared<MultiResponseProbsMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(reponse_memory), 0, reponse_memory->count);

                    for (size_t i = 0; i < x->count; i += m_max_batch_size)
                    {
                        triton::client::InferInput* input1;

                        size_t start = i;
                        size_t stop  = std::min(i + m_max_batch_size, x->count);

                        reader_type_t mini_batch_input =
                            std::static_pointer_cast<MultiInferenceMessage>(x->get_slice(start, stop));
                        writer_type_t mini_batch_output =
                            std::static_pointer_cast<MultiResponseProbsMessage>(response->get_slice(start, stop));

                        // Iterate on the model inputs in case the model takes less than what tensors are available
                        std::vector<std::pair<std::shared_ptr<triton::client::InferInput>, std::vector<uint8_t>>> saved_inputs =
                            foreach_map(m_model_inputs, [this, &mini_batch_input](auto const& model_input) {
                                DCHECK(mini_batch_input->memory->has_input(model_input.mapped_name))
                                    << "Model input '" << model_input.mapped_name << "' not found in InferenceMemory";

                                auto const& inp_tensor = mini_batch_input->get_input(model_input.mapped_name);

                                // Convert to the right type. Make shallow if necessary
                                auto final_tensor = inp_tensor.as_type(model_input.datatype);

                                std::vector<uint8_t> inp_data = final_tensor.get_host_data();

                                // Test
                                triton::client::InferInput* inp_ptr;

                                triton::client::InferInput::Create(&inp_ptr,
                                                       model_input.name,
                                                       {inp_tensor.shape(0), inp_tensor.shape(1)},
                                                       model_input.datatype.triton_str());
                                std::shared_ptr<triton::client::InferInput> inp_shared;
                                inp_shared.reset(inp_ptr);

                                inp_ptr->AppendRaw(inp_data);

                                return std::make_pair(inp_shared, std::move(inp_data));
                            });

                        std::vector<std::shared_ptr<const triton::client::InferRequestedOutput>> saved_outputs =
                            foreach_map(m_model_outputs, [this](auto const& model_output) {
                                // Generate the outputs to be requested.
                                triton::client::InferRequestedOutput* out_ptr;

                                triton::client::InferRequestedOutput::Create(&out_ptr, model_output.name);
                                std::shared_ptr<const triton::client::InferRequestedOutput> out_shared;
                                out_shared.reset(out_ptr);

                                return out_shared;
                            });

                        std::vector<triton::client::InferInput*> inputs =
                            foreach_map(saved_inputs, [](auto x) { return x.first.get(); });

                        std::vector<const triton::client::InferRequestedOutput*> outputs =
                            foreach_map(saved_outputs, [](auto x) { return x.get(); });

                        // this->segment().resources().fiber_pool().enqueue([client, output](){});

                        triton::client::InferResult* results;

                        CHECK_TRITON(client->Infer(&results, m_options, inputs, outputs));

                        for (auto& model_output : m_model_outputs)
                        {
                            std::vector<int64_t> output_shape;

                            CHECK_TRITON(results->Shape(model_output.name, &output_shape));

                            // Make sure we have at least 2 dims
                            while (output_shape.size() < 2)
                            {
                                output_shape.push_back(1);
                            }

                            const uint8_t* output_ptr = nullptr;
                            size_t output_ptr_size    = 0;
                            CHECK_TRITON(results->RawData(model_output.name, &output_ptr, &output_ptr_size));

                            auto output_buffer =
                                std::make_shared<rmm::device_buffer>(output_ptr_size, rmm::cuda_stream_per_thread);

                            NEO_CHECK_CUDA(
                                cudaMemcpy(output_buffer->data(), output_ptr, output_ptr_size, cudaMemcpyHostToDevice));

                            // If we need to do logits, do that here
                            if (m_needs_logits)
                            {
                                size_t element_count =
                                    std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<>());
                                output_buffer = logits(
                                    DevMemInfo{element_count, model_output.datatype.type_id(), output_buffer, 0});
                            }

                            mini_batch_output->set_output(
                                model_output.mapped_name,
                                Tensor::create(std::move(output_buffer),
                                               model_output.datatype,
                                               std::vector<neo::TensorIndex>{static_cast<int>(output_shape[0]),
                                                                             static_cast<int>(output_shape[1])},
                                               std::vector<neo::TensorIndex>{},
                                               0));
                        }
                    }
                    output.on_next(std::move(response));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_model_name;
    std::string m_server_url;
    bool m_force_convert_inputs;
    bool m_use_shared_memory;
    bool m_needs_logits{true};
    std::map<std::string, std::string> m_inout_mapping;

    // Below are settings created during handshake with server
    // std::shared_ptr<triton::client::InferenceServerHttpClient> m_client;
    std::vector<TritonInOut> m_model_inputs;
    std::vector<TritonInOut> m_model_outputs;
    triton::client::InferOptions m_options;
    int m_max_batch_size{-1};
};

}  // namespace morpheus
