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


#include <morpheus/stages/kafka_source.hpp>

#include <morpheus/messages/meta.hpp>
#include <morpheus/utilities/stage_util.hpp>
#include <morpheus/utilities/string_util.hpp>

#include <neo/core/segment.hpp>
#include <pyneo/node.hpp>

#include <boost/fiber/recursive_mutex.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/replace.hpp>
#include <glog/logging.h>
#include <http_client.h>
#include <librdkafka/rdkafkacpp.h>
#include <nvtext/subword_tokenize.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>


#define CHECK_KAFKA(command, expected, msg)                                                                    \
    {                                                                                                          \
        RdKafka::ErrorCode __code = command;                                                                   \
        if (__code != expected)                                                                                \
        {                                                                                                      \
            LOG(ERROR) << msg << ". Received unexpected ErrorCode. Expected: " << #expected << "(" << expected \
                       << "), Received: " << __code << ", Msg: " << RdKafka::err2str(__code);                  \
        }                                                                                                      \
    };

namespace morpheus {
    // Component-private classes.
    // ************ KafkaSourceStage__UnsubscribedException**************//
    class KafkaSourceStage__UnsubscribedException : public std::exception {
    };

    // ************ KafkaSourceStage__Rebalancer *************************//
    class KafkaSourceStage__Rebalancer : public RdKafka::RebalanceCb {
    public:
        KafkaSourceStage__Rebalancer(
                std::function<neo::SharedFuture<bool>(std::vector<std::function<bool()>> &&)> task_launch_fn,
                std::function<int32_t()> batch_timeout_fn,
                std::function<std::size_t()> max_batch_size_fn,
                std::function<std::string(std::string)> display_str_fn,
                std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>> &)> process_fn);

        void rebalance_cb(RdKafka::KafkaConsumer *consumer,
                          RdKafka::ErrorCode err,
                          std::vector<RdKafka::TopicPartition *> &partitions) override;

        void rebalance_loop(RdKafka::KafkaConsumer *consumer);

        bool is_rebalanced();

    private:
        std::vector<std::unique_ptr<RdKafka::Message>> partition_progress_step(RdKafka::Queue *queue) {
            //auto batch_timeout = std::chrono::milliseconds(m_parent.batch_timeout_ms());
            auto batch_timeout = std::chrono::milliseconds(m_batch_timeout_fn());

            size_t msg_count = 0;
            std::vector<std::unique_ptr<RdKafka::Message>> messages;

            auto now = std::chrono::high_resolution_clock::now();
            auto batch_end = now + batch_timeout;

            do {
                auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - now).count();

                DCHECK(remaining_ms >= 0) << "Cant have negative reminaing time";

                std::unique_ptr<RdKafka::Message> msg{queue->consume(std::min(10L, remaining_ms))};

                switch (msg->err()) {
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
            } while (msg_count < m_max_batch_size_fn() && now < batch_end && m_is_rebalanced);

            return std::move(messages);
        }

        bool m_is_rebalanced{false};

        std::function<neo::SharedFuture<bool>(std::vector<std::function<bool()>> &&)> m_task_launcher_fn;
        std::function<int32_t()> m_batch_timeout_fn;
        std::function<std::size_t()> m_max_batch_size_fn;
        std::function<std::string(std::string)> m_display_str_fn;
        std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>> &)> m_process_fn;

        boost::fibers::recursive_mutex m_mutex;
        neo::SharedFuture<bool> m_partition_future;
    };

    KafkaSourceStage__Rebalancer::KafkaSourceStage__Rebalancer(
            std::function<neo::SharedFuture<bool>(std::vector<std::function<bool()>> &&)> task_launch_fn,
            std::function<int32_t()> batch_timeout_fn,
            std::function<std::size_t()> max_batch_size_fn,
            std::function<std::string(std::string)> display_str_fn,
            std::function<bool(std::vector<std::unique_ptr<RdKafka::Message>> &)> process_fn) :
                m_batch_timeout_fn(std::move(batch_timeout_fn)),
                m_max_batch_size_fn(std::move(max_batch_size_fn)),
                m_display_str_fn(std::move(display_str_fn)),
                m_process_fn(std::move(process_fn)) {}

    void KafkaSourceStage__Rebalancer::rebalance_cb(RdKafka::KafkaConsumer *consumer, RdKafka::ErrorCode err,
                                                    std::vector<RdKafka::TopicPartition *> &partitions) {
        std::unique_lock<boost::fibers::recursive_mutex> lock(m_mutex);

        if (err == RdKafka::ERR__ASSIGN_PARTITIONS) {
            VLOG(10) << m_display_str_fn("Rebalance: Assign Partitions");

            // application may load offets from arbitrary external storage here and update \p partitions
            if (consumer->rebalance_protocol() == "COOPERATIVE") {
                CHECK_KAFKA(std::unique_ptr<RdKafka::Error>(consumer->incremental_assign(partitions))->code(),
                            RdKafka::ERR_NO_ERROR,
                            "Error during incremental assign");
            } else {
                CHECK_KAFKA(consumer->assign(partitions), RdKafka::ERR_NO_ERROR, "Error during assign");
            }

            std::vector<std::function<bool()>> tasks;

            for (auto partition: partitions) {
                auto queue_ptr = consumer->get_partition_queue(partition);

                // Now forward to one of the running queues
                // queue->forward(m_parent.m_queues[i % m_parent.m_queues.size()].get());
                queue_ptr->forward(nullptr);

                auto partition_ptr = RdKafka::TopicPartition::create(
                        partition->topic(), partition->partition(), partition->offset());

                tasks.emplace_back([q = queue_ptr, p = partition_ptr, consumer, this]() {
                    auto partition = std::unique_ptr<RdKafka::TopicPartition>(p);
                    auto queue = std::unique_ptr<RdKafka::Queue>(q);

                    while (m_is_rebalanced) {
                        // Build the batch
                        auto messages = this->partition_progress_step(queue.get());

                        try {
                            // Process the messages. Returns true if we need to commit
                            auto should_commit = m_process_fn(messages);

                            if (should_commit) {
                                int64_t max_offset = -1000;
                                for (auto &m: messages) {
                                    DCHECK(m->partition() == partition->partition())
                                                    << "Inconsistent error. Message partition does not match fiber partition";

                                    max_offset = std::max(max_offset, m->offset());
                                }

                                // Find the last message for this partition
                                partition->set_offset(max_offset + 1);

                                CHECK_KAFKA(
                                        consumer->commitAsync(
                                                std::vector<RdKafka::TopicPartition *>{partition.get()}),
                                        RdKafka::ERR_NO_ERROR,
                                        "Error during commitAsync");
                            }
                        } catch (KafkaSourceStage__UnsubscribedException &) {
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

            m_partition_future = std::move(m_task_launcher_fn(std::move(tasks)));
        } else if (err == RdKafka::ERR__REVOKE_PARTITIONS) {
            VLOG(10) << m_display_str_fn("Rebalance: Revoke Partitions");

            // Application may commit offsets manually here if auto.commit.enable=false
            if (consumer->rebalance_protocol() == "COOPERATIVE") {
                CHECK_KAFKA(std::unique_ptr<RdKafka::Error>(consumer->incremental_unassign(partitions))->code(),
                            RdKafka::ERR_NO_ERROR,
                            "Error during incremental unassign");
            } else {
                CHECK_KAFKA(consumer->unassign(), RdKafka::ERR_NO_ERROR, "Error during unassign");
            }

            // Stop all processing queues
            m_is_rebalanced = false;

            // Wait until all processing has completed
            if (m_partition_future.valid()) {
                m_partition_future.wait();
            }
        } else {
            LOG(ERROR) << "Rebalancing error: " << RdKafka::err2str(err) << std::endl;
            CHECK_KAFKA(consumer->unassign(), RdKafka::ERR_NO_ERROR, "Error during unassign");
        }
    }

    void KafkaSourceStage__Rebalancer::rebalance_loop(RdKafka::KafkaConsumer *consumer) {
        do {
            // Poll until we are rebalanced
            while (!this->is_rebalanced()) {
                VLOG(10) << m_display_str_fn("Rebalance: Calling poll to trigger rebalance");
                consumer->poll(500);
            }
        } while (m_partition_future.get());
    }

    bool KafkaSourceStage__Rebalancer::is_rebalanced() {
        std::unique_lock<boost::fibers::recursive_mutex> lock(m_mutex);

        return m_is_rebalanced;
    }

    // Component public implementations
    // ************ KafkaStage ************************* //
    KafkaSourceStage::KafkaSourceStage(const neo::Segment &parent,
                                       const std::string &name,
                                       std::size_t max_batch_size,
                                       std::string topic,
                                       int32_t batch_timeout_ms,
                                       std::map<std::string, std::string> config,
                                       bool disable_commit) :
            neo::SegmentObject(parent, name),
            base_t(parent, name),
            m_max_batch_size(max_batch_size),
            m_topic(std::move(topic)),
            m_batch_timeout_ms(batch_timeout_ms),
            m_config(std::move(config)),
            m_disable_commit(disable_commit) {
        this->set_source_observable(neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t> &sub) {
            // Build rebalancer
            KafkaSourceStage__Rebalancer rebalancer([this](std::vector<std::function<bool()>> &&tasks) {
                                                        return this->launch_tasks(std::move(tasks));
                                                    },
                                                    [this]() { return this->batch_timeout_ms(); },
                                                    [this]() { return this->max_batch_size(); },
                                                    [this](const std::string str_to_display) {
                                                        return this->display_str(str_to_display);
                                                    },
                                                    [&sub, this](
                                                            std::vector<std::unique_ptr<RdKafka::Message>> &message_batch) {
                                                        // If we are unsubscribed, throw an error to break the loops
                                                        if (!sub.is_subscribed()) {
                                                            throw KafkaSourceStage__UnsubscribedException();
                                                        }

                                                        if (!message_batch.empty()) {
                                                            auto batch = this->process_batch(std::move(message_batch));

                                                            sub.on_next(std::move(batch));

                                                            if (m_requires_commit) {
                                                                return true;
                                                            }
                                                        }

                                                        return false;
                                                    });

            // Build consumer
            m_rebalancer = &rebalancer;
            auto consumer = this->create_consumer();

            rebalancer.rebalance_loop(consumer.get());

            consumer->unsubscribe();

            consumer->close();

            consumer.reset();

            m_rebalancer = nullptr;

            sub.on_completed();
        }));
    }

    std::size_t KafkaSourceStage::max_batch_size() {
        return m_max_batch_size;
    }

    int32_t KafkaSourceStage::batch_timeout_ms() {
        return m_batch_timeout_ms;
    }

    void KafkaSourceStage::start() {
        // Save off the queues before setting our concurrency back to 1
        for (size_t i = 0; i < this->concurrency(); ++i) {
            m_task_queues.push_back(this->resources().fiber_pool().next_task_queue());
        }

        this->concurrency(1);

        // Call the default start
        neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>::start();
    }

    std::unique_ptr<RdKafka::Conf>
    KafkaSourceStage::build_kafka_conf(const std::map<std::string, std::string> &config_in) {
        // Copy the config
        std::map<std::string, std::string> config_out(config_in);

        std::map<std::string, std::string> defaults{{"session.timeout.ms",   "60000"},
                                                    {"enable.auto.commit",   "false"},
                                                    {"auto.offset.reset",    "latest"},
                                                    {"enable.partition.eof", "true"}};

        // Set some defaults if they dont exist
        config_out.merge(defaults);

        m_requires_commit = config_out["enable.auto.commit"] == "false";

        if (m_requires_commit && m_disable_commit) {
            LOG(WARNING)
                    << "KafkaSourceStage: Commits have been disabled for this Kafka consumer. This should only be "
                       "used in a debug environment";
            m_requires_commit = false;
        } else if (!m_requires_commit && m_disable_commit) {
            // User has auto-commit on and disable commit at same time
            LOG(WARNING)
                    << "KafkaSourceStage: The config option 'enable.auto.commit' was set to True but commits have "
                       "been disabled for this Kafka consumer. This should only be used in a debug environment";
        }

        // Make the kafka_conf and set all properties
        auto kafka_conf = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

        for (auto const &key_value: config_out) {
            std::string error_string;
            if (RdKafka::Conf::ConfResult::CONF_OK !=
                kafka_conf->set(key_value.first, key_value.second, error_string)) {
                LOG(ERROR) << "Error occurred while setting Kafka configuration. Error: " << error_string;
            }
        }

        return std::move(kafka_conf);
    }

    neo::SharedFuture<bool> KafkaSourceStage::launch_tasks(std::vector<std::function<bool()>> &&tasks) {
        std::vector<neo::SharedFuture<bool>>
                partition_futures;

        // Loop over tasks enqueuing onto saved fiber queues
        for (size_t i = 0; i < tasks.size(); ++i) {
            partition_futures.emplace_back(
                    this->m_task_queues[i % this->m_task_queues.size()]->enqueue(std::move(tasks[i])));
        }

        // Launch a new task to await on the futures
        return this->m_task_queues[tasks.size() % this->m_task_queues.size()]->enqueue([partition_futures]() {
            bool ret_val = true;

            for (auto &f: partition_futures) {
                if (!f.get()) {
                    // Return false if any return false
                    ret_val = false;
                }
            }

            return ret_val;
        });
    }

    std::unique_ptr<RdKafka::KafkaConsumer>
    KafkaSourceStage::create_consumer() {
        auto rebalancer = static_cast<KafkaSourceStage__Rebalancer *>(m_rebalancer);
        auto kafka_conf = this->build_kafka_conf(m_config);
        std::string errstr;

        if (RdKafka::Conf::ConfResult::CONF_OK != kafka_conf->set("rebalance_cb", rebalancer, errstr)) {
            LOG(FATAL) << "Error occurred while setting Kafka rebalance function. Error: " << errstr;
        }

        auto consumer =
                std::unique_ptr<RdKafka::KafkaConsumer>(RdKafka::KafkaConsumer::create(kafka_conf.get(), errstr));

        if (!consumer) {
            LOG(FATAL) << "Error occurred creating Kafka consumer. Error: " << errstr;
        }

        // Subscribe to the topic. Uses the default rebalancer
        CHECK_KAFKA(consumer->subscribe(std::vector<std::string>{m_topic}),
                    RdKafka::ERR_NO_ERROR,
                    "Error subscribing to topics");

        auto spec_topic =
                std::unique_ptr<RdKafka::Topic>(RdKafka::Topic::create(consumer.get(), m_topic, nullptr, errstr));

        RdKafka::Metadata *md;
        CHECK_KAFKA(consumer->metadata(spec_topic == nullptr, spec_topic.get(), &md, 1000),
                    RdKafka::ERR_NO_ERROR,
                    "Failed to list_topics in Kafka broker");

        std::map<std::string, std::vector<int32_t>> topic_parts;

        VLOG(10) << this->display_str(CONCAT_STR("Subscribed to " << md->topics()->size() << " topics:"));

        for (auto const &topic: *(md->topics())) {
            auto &part_ids = topic_parts[topic->topic()];

            auto const &parts = *(topic->partitions());

            std::transform(parts.cbegin(), parts.cend(), std::back_inserter(part_ids), [](auto const &part) {
                return part->id();
            });

            auto toppar_list = foreach_map(parts, [&topic](const auto &part) {
                return std::unique_ptr<RdKafka::TopicPartition>{
                        RdKafka::TopicPartition::create(topic->topic(), part->id())};
            });

            std::vector<RdKafka::TopicPartition *> toppar_ptrs =
                    foreach_map(toppar_list,
                                [](const std::unique_ptr<RdKafka::TopicPartition> &x) { return x.get(); });

            // Query Kafka to populate the TopicPartitions with the desired offsets
            CHECK_KAFKA(consumer->committed(toppar_ptrs, 1000),
                        RdKafka::ERR_NO_ERROR,
                        "Failed retrieve Kafka committed offsets");

            auto committed =
                    foreach_map(toppar_list,
                                [](const std::unique_ptr<RdKafka::TopicPartition> &x) { return x->offset(); });

            // Query Kafka to populate the TopicPartitions with the desired offsets
            CHECK_KAFKA(consumer->position(toppar_ptrs), RdKafka::ERR_NO_ERROR, "Failed retrieve Kafka positions");

            auto positions =
                    foreach_map(toppar_list,
                                [](const std::unique_ptr<RdKafka::TopicPartition> &x) { return x->offset(); });

            VLOG(10) << this->display_str(CONCAT_STR(
                                                  "   Topic: '" << topic->topic() << "', Parts: "
                                                                << StringUtil::array_to_str(part_ids.begin(),
                                                                                            part_ids.end())
                                                                << ", Committed: "
                                                                << StringUtil::array_to_str(committed.begin(),
                                                                                            committed.end())
                                                                << ", Positions: "
                                                                << StringUtil::array_to_str(positions.begin(),
                                                                                            positions.end())));
        }

        return std::move(consumer);
    }

    cudf::io::table_with_metadata KafkaSourceStage::load_table(const std::string &buffer) {
        auto options =
                cudf::io::json_reader_options::builder(cudf::io::source_info(buffer.c_str(), buffer.size())).lines(
                        true);

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
                cudf::strings_column_view{columns[idx]->view()}, cudf::string_scalar("\\n"),
                cudf::string_scalar("\n"));

        updated_data = cudf::strings::replace(
                cudf::strings_column_view{updated_data->view()}, cudf::string_scalar("\\/"),
                cudf::string_scalar("/"));

        columns[idx] = std::move(updated_data);

        tbl.tbl = std::move(std::make_unique<cudf::table>(std::move(columns)));

        return tbl;
    }

    std::shared_ptr<morpheus::MessageMeta>
    KafkaSourceStage::process_batch(std::vector<std::unique_ptr<RdKafka::Message>> &&message_batch) {
        std::ostringstream buffer;

        // Build the string for the batch
        for (auto &msg: message_batch) {
            buffer << static_cast<char *>(msg->payload()) << "\n";
        }

        auto data_table = this->load_table(buffer.str());

        // Next, create the message metadata. This gets reused for repeats
        auto meta = MessageMeta::create_from_cpp(std::move(data_table), 0);

        return meta;
    }

    // ************ KafkaStageInterfaceProxy ************ //
    std::shared_ptr<KafkaSourceStage>
    KafkaSourceStageInterfaceProxy::init(neo::Segment &parent, const std::string &name, size_t max_batch_size,
                                         std::string topic, int32_t batch_timeout_ms,
                                         std::map<std::string, std::string> config, bool disable_commits) {
        auto stage = std::make_shared<KafkaSourceStage>(
                parent, name, max_batch_size, topic, batch_timeout_ms, config, disable_commits);

        parent.register_node<KafkaSourceStage>(stage);

        return stage;
    }
}
