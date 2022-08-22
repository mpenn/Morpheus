# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import logging
import time
from datetime import timedelta
from io import StringIO

import pika
import srf

import cudf

from _lib import morpheus_rabbit as morpheus_rabbit_cpp
from morpheus.config import Config
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class RabbitMQSourceStage(SingleOutputSource):
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    host : str
        Hostname or IP of the RabbitMQ server.
    exchange : str
        Name of the RabbitMQ exchange to connect to.
    exchange_type : str
        RabbitMQ exchange type; defaults to `fanout`.
    queue_name : str
        Name of the queue to listen to. If left blank, RabbitMQ will generate a random queue name bound to the exchange.
    poll_interval : timedelta
        Amount of time  between polling RabbitMQ for new messages; defaults to 100ms
    """

    def __init__(self,
                 config: Config,
                 host: str,
                 exchange: str,
                 exchange_type: str = 'fanout',
                 queue_name: str = '',
                 poll_interval: timedelta = None):
        super().__init__(config)
        self._host = host
        self._exchange = exchange
        self._exchange_type = exchange_type
        self._queue_name = queue_name

        self._connection = None
        self._channel = None

        if poll_interval is not None:
            self._poll_interval = poll_interval
        else:
            self._poll_interval = timedelta(milliseconds=100)

        # Flag to indicate whether or not we should stop
        self._stop_requested = False

    @property
    def name(self) -> str:
        return "from-rabbitmq"

    @classmethod
    def supports_cpp_node(cls) -> bool:
        return True

    def _build_source(self, builder: srf.Builder) -> StreamPair:
        if self._build_cpp_node():
            node = morpheus_rabbit_cpp.RabbitMQSourceStage(builder,
                                                           self.unique_name,
                                                           self._host,
                                                           self._exchange,
                                                           self._exchange_type,
                                                           self._queue_name,
                                                           self._poll_interval)
        else:
            self.connect()
            node = builder.make_source(self.unique_name, self.source_generator)
        return node, MessageMeta

    def connect(self):
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=self._host))

        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange=self._exchange, exchange_type=self._exchange_type)

        result = self._channel.queue_declare(queue=self._queue_name, exclusive=True)

        # When queue_name='' we will receive a randomly generated queue name
        self._queue_name = result.method.queue

        self._channel.queue_bind(exchange=self._exchange, queue=self._queue_name)

    def source_generator(self):
        try:
            while not self._stop_requested:
                (method_frame, header_frame, body) = self._channel.basic_get(self._queue_name)
                if method_frame is not None:
                    try:
                        buffer = StringIO(body.decode("utf-8"))
                        df = cudf.io.read_json(buffer, orient='records', lines=True)
                        yield MessageMeta(df=df)
                    except Exception as ex:
                        logger.exception("Error occurred converting RabbitMQ message to Dataframe: {}".format(ex))
                    finally:
                        self._channel.basic_ack(method_frame.delivery_tag)
                else:
                    # queue is empty, sleep before polling again
                    time.sleep(self._poll_interval.total_seconds())

        finally:
            self._connection.close()
