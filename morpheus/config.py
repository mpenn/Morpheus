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

"""
Contains configuration objects used to run pipeline and utilities.
"""

import dataclasses
import json
import typing
import logging

from enum import Enum

logger = logging.getLogger(__name__)

def auto_determine_bootstrap():
    """Auto determine bootstrap servers for kafka cluster."""
    import docker

    kafka_compose_name = "kafka-docker"

    docker_client = docker.from_env()
    bridge_net = docker_client.networks.get("bridge")
    bridge_ip = bridge_net.attrs["IPAM"]["Config"][0]["Gateway"]

    kafka_net = docker_client.networks.get(kafka_compose_name + "_default")

    bootstrap_servers = ",".join([
        c.ports["9092/tcp"][0]["HostIp"] + ":" + c.ports["9092/tcp"][0]["HostPort"] for c in kafka_net.containers
        if "9092/tcp" in c.ports
    ])

    # Use this version to specify the bridge IP instead
    bootstrap_servers = ",".join(
        [bridge_ip + ":" + c.ports["9092/tcp"][0]["HostPort"] for c in kafka_net.containers if "9092/tcp" in c.ports])

    logger.info("Auto determined Bootstrap Servers: {}".format(bootstrap_servers))

    return bootstrap_servers


class ConfigWrapper(object):
    """
    Config wrapper class

    Parameters
    ----------
    internal_dict : dict
        internal dictionary

    """
    def __init__(self, internal_dict: dict):

        self._state = internal_dict

    def __getattr__(self, name):

        if name in self._state:

            if isinstance(self._state[name], dict):
                return ConfigWrapper(self._state[name])
            else:
                return self._state[name]
        elif hasattr(self._state, name):
            return self._state.__getattribute__(name)

        # Otherwise return error
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __getitem__(self, key):

        if key in self._state:
            return self._state[key]

        # Otherwise return error
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, key))

    def __setitem__(self, key, value):

        # Cant add new keys
        if key in self._state:
            self._state[key] = value
            return

        # Otherwise return error
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, key))

    def __contains__(self, key):

        return self._state.__contains__(key)


@dataclasses.dataclass
class ConfigBase():
    """This is the base class for pipeline configuration."""
    pass

@dataclasses.dataclass
class ConfigOnnxToTRT(ConfigBase):
    """
    Configuration class for the OnnxToTRT migration process.

    Parameters
    ----------
    input_model : str
        Input model file.
    output_model : str
        Output model file.
    batches : typing.List[typing.Tuple[int, int]]
        Batches of input data
    seq_length : int
        Sequence length
    max_workspace_size : int
        Maximum workspace size, default 16000 MB

    """
    input_model: str = None
    output_model: str = None
    batches: typing.List[typing.Tuple[int, int]] = dataclasses.field(default_factory=list)
    seq_length: int = None
    max_workspace_size: int = 16000  # In MB


@dataclasses.dataclass
class ConfigGeneral(ConfigBase):
    """
    General pipeline configuration class.

    Parameters
    ----------
    debug : bool
        Debug flag to run pipeline in debug mode, default is set False.
    pipeline : str
        Pipeline name.

    """
    debug = False
    pipeline: str = "pytorch"


@dataclasses.dataclass
class ConfigModel(ConfigBase):
    """
    Pipeline model configuration class.

    Parameters
    ----------
    vocab_hash_file : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw
        vocabulary using the cudf.utils.hash_vocab_utils.hash_vocab function. Default value is
        'bert-base-cased-hash.txt'
    seq_length : int
        Limits the length of the sequence returned. If tokenized string is shorter than max_length, output
        will be padded with 0s. If the tokenized string is longer than max_length and do_truncate == False,
        there will be multiple returned sequences containing the overflowing token-ids. Default value is 128
    max_batch_size : int
        Model max batch size. Default value is 8

    """
    vocab_hash_file: str = "bert-base-cased-hash.txt"
    seq_length: int = 128
    max_batch_size: int = 8


@dataclasses.dataclass
class ConfigKafka(ConfigBase):
    """
    Pipeline Kafka configuration class.

    Parameters
    ----------
    bootstrap_servers : str
        Bootstrap server hostname and port number to connect Kafka cluster.
    input_topic : str
        Input Kafka topic to consume messages
    output_topic : str
        Output Kafka topic to publish messages

    """
    bootstrap_servers: str = "auto"
    input_topic: str = "test_pcap"
    output_topic: str = "output_topic"


@dataclasses.dataclass
class ConfigDask(ConfigBase):
    """
    Pipeline Dask configuration class.

    Parameters
    ----------
    use_processes : bool
        This parameter determines whether pipeline should be run in dask mode.

    """
    use_processes: bool = False

@dataclasses.dataclass
class ConfigNlp(ConfigBase):
    """
    NLP category pipeline configuration class.

    Parameters
    ----------
    model_vocab_hash_file : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw
        vocabulary using the cudf.utils.hash_vocab_utils.hash_vocab function. Default value is
        'data/bert-base-cased-hash.txt'
    model_seq_length : int
        Limits the length of the sequence returned. If tokenized string is shorter than max_length, output
        will be padded with 0s. If the tokenized string is longer than max_length and do_truncate == False,
        there will be multiple returned sequences containing the overflowing token-ids. Default value is 256

    """
    model_vocab_hash_file: str = "data/bert-base-cased-hash.txt"
    model_seq_length: int = 256


@dataclasses.dataclass
class ConfigFil(ConfigBase):
    """
    FIL category pipeline configuration class.

    Parameters
    ----------
    model_fea_length : int
        The number of features used to infer a model. Default value is 29

    """
    model_fea_length: int = 29

class PipelineModes(str, Enum):
    """The type of usecases that can be executed by the pipeline is determined by the enum."""
    NLP = "NLP"
    FIL = "FIL"

@dataclasses.dataclass
class Config(ConfigBase):
    """
    Pipeline configuration class.

    Parameters
    ----------
    debug : bool
        Flag to run pipeline in debug mode. Default value is False
    mode : PipelineModes
        Determines type of pipeline Ex: FIL or NLP
    pipeline_batch_size : int
        Determines number of messages per batch. Default value is 256
    num_threads : int
        Number threads to process each batch of messages. Default value is 1
    model_max_batch_size : 8
        In a single batch, the maximum number of messages to send to the model for inference. Default value is
        8
    use_dask :
        Determines if the pipeline should be executed using the Dask scheduler. Default value is False
    nlp : morpheus.config.ConfigNlp
        NLP category models pipeline configuration.
    fil : morpheus.config.ConfigFil
        FIL category models pipeline configuration.

    """
    # Flag to indicate we are creating a static instance. Prevents raising an error on creation
    __is_creating: typing.ClassVar[bool] = False

    __default: typing.ClassVar["Config"] = None
    __instance: typing.ClassVar["Config"] = None

    debug: bool = False
    log_level: int = logging.WARN
    log_config_file: str = None

    mode: PipelineModes = None

    pipeline_batch_size: int = 256
    num_threads: int = 1
    model_max_batch_size: int = 8
    
    use_dask: bool = False

    dask: ConfigDask = dataclasses.field(default_factory=ConfigDask)

    nlp: ConfigNlp = dataclasses.field(default_factory=ConfigNlp)
    fil: ConfigFil = dataclasses.field(default_factory=ConfigFil)

    @property
    def feature_length(self):
        return self.nlp.model_seq_length if self.mode == PipelineModes.NLP else self.fil.model_fea_length

    @staticmethod
    def default() -> "Config":
        if Config.__default is None:
            try:
                Config.__is_creating = True
                Config.__default = Config()
            finally:
                Config.__is_creating = False

        return Config.__default

    @classmethod
    def get(cls) -> "Config":
        if cls.__instance is None:
            try:
                cls.__is_creating = True
                cls.__instance = Config()
            finally:
                cls.__is_creating = False

        return cls.__instance

    def __post_init__(self):
        # Double check that this class is not being created outside of .get()
        if not Config.__is_creating:
            raise Exception("This class is a singleton! Use Config.default() or Config.get() for instances")

    def load(self, filename: str):
        # Read the json file and store as
        raise NotImplementedError("load() has not been implemented yet.")

    def save(self, filename: str):
        # Read the json file and store as
        with open(filename, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=3, sort_keys=True)

    def to_string(self):
        # pp = pprint.PrettyPrinter(indent=2, width=80)

        # return pp.pformat(dataclasses.asdict(self))

        # Using JSON serializer for now since its easier to read. pprint is more compact
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


