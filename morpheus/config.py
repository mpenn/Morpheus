import dataclasses
import json
import pprint
import typing

import docker


def auto_determine_bootstrap():
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

    print("Auto determined Bootstrap Servers: {}".format(bootstrap_servers))

    return bootstrap_servers


class ConfigWrapper(object):
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
class ConfigOnnxToTRT():
    input_model: str = None
    output_model: str = None
    batches: typing.List[typing.Tuple[int, int]] = dataclasses.field(default_factory=list)
    seq_length: int = None
    max_workspace_size: int = 16000  # In MB


@dataclasses.dataclass
class ConfigGeneral():
    debug = False
    pipeline: str = "pytorch"


@dataclasses.dataclass
class ConfigModel():
    vocab_hash_file: str = "bert-base-cased-hash.txt"
    seq_length: int = 128
    max_batch_size: int = 8


@dataclasses.dataclass
class ConfigKafka():
    bootstrap_servers: str = "auto"
    input_topic: str = "test_pcap"
    output_topic: str = "output_topic"


@dataclasses.dataclass
class ConfigDask():
    use_processes: bool = False


@dataclasses.dataclass
class Config():

    # Flag to indicate we are creating a static instance. Prevents raising an error on creation
    __is_creating: typing.ClassVar[bool] = False

    __default: typing.ClassVar["Config"] = None
    __instance: typing.ClassVar["Config"] = None

    pipeline_batch_size: int = 256
    num_threads: int = 1
    model_max_batch_size: int = 8
    model_seq_length: int = 256
    model_vocab_hash_file: str = "bert-base-cased-hash.txt"
    use_dask: bool = False

    dask: ConfigDask = dataclasses.field(default_factory=ConfigDask)

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
        pp = pprint.PrettyPrinter(indent=2, width=80)

        return pp.pformat(dataclasses.asdict(self))
