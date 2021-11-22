# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import typing
import weakref

import neo
from custreamz import kafka
from streamz.core import Stream
from tornado.ioloop import IOLoop

import cudf

from morpheus.config import Config
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair


class KafkaSourceStage(SingleOutputSource):
    """
    Load messages from a Kafka cluster.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    bootstrap_servers : str
        Kafka cluster bootstrap servers separated by a comma.
    input_topic : str
        Input kafka topic
    group_id : str
        Specifies the name of the consumer group a Kafka consumer belongs to
    use_dask : bool
        Determines whether or not dask should be used to consume messages. Operates independently of the
        `Pipeline.use_dask` option
    poll_interval : str
        Seconds that elapse between polling Kafka for new messages. Follows the pandas interval format

    """
    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: str = "test_pcap",
                 group_id: str = "custreamz",
                 use_dask: bool = False,
                 poll_interval: str = "10millis"):
        super().__init__(c)

        self._consumer_conf = {
            'bootstrap.servers': bootstrap_servers, 'group.id': group_id, 'session.timeout.ms': "60000"
        }

        self._input_topic = input_topic
        self._use_dask = use_dask
        self._poll_interval = poll_interval
        self._max_batch_size = c.pipeline_batch_size
        self._max_concurrent = c.num_threads
        self._client = None

    @property
    def name(self) -> str:
        return "from-kafka"

    def _source_generator(self, s: neo.Subscriber):
        # Each invocation of this function makes a new thread so recreate the producers

        # Set some initial values
        npartitions = self._npartitions
        consumer = None
        consumer_params = self._consumer_params
        # Override the auto-commit config to enforce custom streamz checkpointing
        consumer_params['enable.auto.commit'] = 'false'
        if 'auto.offset.reset' not in consumer_params.keys():
            consumer_params['auto.offset.reset'] = 'latest'

        # Now begin the script
        import confluent_kafka as ck

        if self.engine == "cudf":  # pragma: no cover
            from custreamz import kafka

        if self.engine == "cudf":  # pragma: no cover
            consumer = kafka.Consumer(consumer_params)
        else:
            consumer = ck.Consumer(consumer_params)

        # weakref.finalize(self, lambda c=consumer: _close_consumer(c))
        tp = ck.TopicPartition(self.topic, 0, 0)

        # blocks for consumer thread to come up
        consumer.get_watermark_offsets(tp)

        if npartitions is None:

            kafka_cluster_metadata = consumer.list_topics(self.topic)

            if self.engine == "cudf":  # pragma: no cover
                npartitions = len(kafka_cluster_metadata[self.topic.encode('utf-8')])
            else:
                npartitions = len(kafka_cluster_metadata.topics[self.topic].partitions)

        positions = [0] * npartitions

        tps = []
        for partition in range(npartitions):
            tps.append(ck.TopicPartition(self.topic, partition))

        while s.is_subscribed():
            try:
                committed = consumer.committed(tps, timeout=1)
            except ck.KafkaException:
                pass
            else:
                for tp in committed:
                    positions[tp.partition] = tp.offset
                break

        while s.is_subscribed():
            out = []

            if self.refresh_partitions:
                kafka_cluster_metadata = consumer.list_topics(self.topic)

                if self.engine == "cudf":  # pragma: no cover
                    new_partitions = len(kafka_cluster_metadata[self.topic.encode('utf-8')])
                else:
                    new_partitions = len(kafka_cluster_metadata.topics[self.topic].partitions)

                if new_partitions > npartitions:
                    positions.extend([-1001] * (new_partitions - npartitions))
                    npartitions = new_partitions

            for partition in range(npartitions):

                tp = ck.TopicPartition(self.topic, partition, 0)

                try:
                    low, high = consumer.get_watermark_offsets(tp, timeout=0.1)
                except (RuntimeError, ck.KafkaException):
                    continue

                self.started = True

                if 'auto.offset.reset' in consumer_params.keys():
                    if consumer_params['auto.offset.reset'] == 'latest' and positions[partition] == -1001:
                        positions[partition] = high

                current_position = positions[partition]

                lowest = max(current_position, low)

                if high > lowest + self.max_batch_size:
                    high = lowest + self.max_batch_size
                if high > lowest:
                    out.append((consumer_params, self.topic, partition, self.keys, lowest, high - 1))
                    positions[partition] = high

            consumer_params['auto.offset.reset'] = 'earliest'

            if (out):
                for part in out:

                    def commit():
                        topic, part_no, _, _, offset = part[1:]
                        _tp = ck.TopicPartition(topic, part_no, offset + 1)
                        consumer.commit(offsets=[_tp], asynchronous=True)

                    weakref.finalize(self, lambda: commit())

                    # Create a future so we can add a done callback to commit the message
                    # Finally push the message
                    s.on_next(part)
            else:
                time.sleep(self.poll_interval)

        s.on_completed()

    def _build_source(self, seg: neo.Segment) -> StreamPair:

        seg.make_source(self.unique_name, self._source_fn())

        if (self._use_dask):
            from dask.distributed import Client
            self._client = Client()

            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=True,
                                                       engine="cudf",
                                                       poll_interval=self._poll_interval,
                                                       max_concurrent=self._max_concurrent,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)

            return source, StreamFuture[cudf.DataFrame]
        else:
            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=False,
                                                       engine="cudf",
                                                       poll_interval=self._poll_interval,
                                                       max_concurrent=self._max_concurrent,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)

            return source, cudf.DataFrame
