from morpheus.pipeline.pipeline import Stage
import typing
from streamz.core import Stream
from tornado.ioloop import IOLoop
from morpheus.pipeline import SourceStage
from morpheus.config import Config

class WriteToKafkaStage(Stage):
    def __init__(self, c: Config, bootstrap_servers: str, output_topic: str):
        super().__init__(c)

        self._kafka_conf = {'bootstrap.servers': bootstrap_servers}

        self._output_topic = output_topic

    @property
    def name(self) -> str:
        return "write_kafka"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        # Write to kafka
        input_stream[0].to_kafka(self._output_topic, self._kafka_conf)

        # Return input unchanged
        return input_stream
