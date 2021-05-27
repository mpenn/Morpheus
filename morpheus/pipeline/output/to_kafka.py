import typing

import typing_utils

from morpheus.config import Config
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair


class WriteToKafkaStage(SinglePortStage):
    """
    Write messages to a Kafka cluster.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    bootstrap_servers : str
        Kafka cluster bootstrap servers separated by comma
    output_topic : str
        Output kafka topic

    """
    def __init__(self, c: Config, bootstrap_servers: str, output_topic: str):
        super().__init__(c)

        self._kafka_conf = {'bootstrap.servers': bootstrap_servers}

        self._output_topic = output_topic

    @property
    def name(self) -> str:
        return "to-kafka"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[list[str], ]
            Accepted input types

        """
        return (typing.List[str], )

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        stream = input_stream[0]
        input_type = input_stream[1]

        # Gather just in case we are using dask
        stream = stream.gather()

        if (typing_utils.issubtype(input_type, typing.Iterable)):
            stream = stream.flatten()

        # Write to kafka
        stream = stream.to_kafka(self._output_topic, self._kafka_conf)

        # Return input unchanged
        return input_stream
