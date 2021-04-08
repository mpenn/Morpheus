import typing_utils
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.pipeline import Stage, StreamPair
import typing
from streamz.core import Stream
from tornado.ioloop import IOLoop
from morpheus.pipeline import SourceStage
from morpheus.config import Config
# from kafka import KafkaProducer
# from kafka.producer.future import FutureRecordMetadata


class WriteToKafkaStage(Stage):
    def __init__(self, c: Config, bootstrap_servers: str, output_topic: str):
        super().__init__(c)

        self._kafka_conf = {'bootstrap.servers': bootstrap_servers}

        self._output_topic = output_topic

    @property
    def name(self) -> str:
        return "to-kafka"

    def accepted_types(self) -> typing.Tuple:
        return (typing.List[str], )

    # def _push_to_kafka(self, x: str):

    #     fut: FutureRecordMetadata = self._producer.send(self._output_topic, x.encode("UTF-8"))

    #     result = fut.get()

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        stream = input_stream[0]
        input_type = input_stream[1]

        # Gather just in case we are using dask
        stream = stream.gather()

        if (typing_utils.issubtype(input_type, typing.Iterable)):
            stream = stream.flatten()

        # self._producer = KafkaProducer(bootstrap_servers=self._kafka_conf["bootstrap.servers"],
        #                                acks=0,
        #                                max_block_ms=4,
        #                                batch_size=16,
        #                                linger_ms=4)

        # stream.sink(self._push_to_kafka)

        # Write to kafka
        stream = stream.to_kafka(self._output_topic, self._kafka_conf)

        # Return input unchanged
        return input_stream
