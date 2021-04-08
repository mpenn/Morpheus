import argparse
import json
import click
import os
import docker
import torch
import typing
import dataclasses
import pprint

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
class Config():

    # Flag to indicate we are creating a static instance. Prevents raising an error on creation
    __is_creating: typing.ClassVar[bool] = False

    __default: typing.ClassVar["Config"] = None
    __instance: typing.ClassVar["Config"] = None

    # general: ConfigGeneral = dataclasses.field(default_factory=ConfigGeneral)
    # model: ConfigModel = dataclasses.field(default_factory=ConfigModel)
    # kafka: ConfigKafka = dataclasses.field(default_factory=ConfigKafka)

    pipeline_batch_size: int = 256
    num_threads: int = 1
    model_max_batch_size: int = 8
    model_seq_length: int = 256
    model_vocab_hash_file: str = "bert-base-cased-hash.txt"

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

    # def __getattr__(self, name):

    #    if name in self._state:
    #       return self._state[name]

    #    # Otherwise return error
    #    raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    # def __delattr__(self, name):
    #    if name in self._state:
    #       del self._state[name]
    #    else:
    #       object.__delattr__(self, name)

    def load(self, filename: str):
        # Read the json file and store as
        raise NotImplementedError("load() has not been implemented yet.")

        # TODO: Implemente loading from dict. See https://stackoverflow.com/a/53498623/634820
        with open(filename) as f:
            state_dict = json.load(f)

            self.general = dataclasses.replace(self.general, **state_dict.general)
            self.model = dataclasses.replace(self.model, **state_dict.model)
            self.kafka = dataclasses.replace(self.kafka, **state_dict.kafka)

    def save(self, filename: str):
        # Read the json file and store as
        with open(filename, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=3, sort_keys=True)

    # def load_preset_dataset(self, preset_name: str):

    #    if preset_name in self._state["dataset"]["presets"]:

    #       ds = self._state["dataset"]
    #       preset = self._state["dataset"]["presets"][preset_name]

    #       for param in ds:
    #          if param in preset:
    #             ds[param] = preset[param]
    #    else:
    #       # Otherwise return error
    #       raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, preset_name))

    # def pre_action(self, action_name: str):

    #    # If description is null, set that automatically
    #    if self._state["general"]["desc"] is None or len(self._state["general"]["desc"]) == 0:
    #       raise click.BadOptionUsage("desc", "Must set description to identify run from others")
    #       # self._state["general"]["desc"] = "training"

    #    # Find the next run idx if not set
    #    run_idx, desc = misc.find_next_result_dir(self.general["logs_dir"], self.general["run_idx"])

    #    gpu_count = min(torch.cuda.device_count(),
    #                    self._state["general"]["gpu_count"]) if self._state["general"]["gpu_count"] is not None else torch.cuda.device_count()

    #    self.general["run_idx"] = run_idx
    #    self.general["gpu_count"] = gpu_count

    #    # fill out the schedule
    #    self._complete_schedule()

    #    # If desc is not none, then it comes from using a previous run again
    #    if desc is not None:
    #       self.general["desc"] = desc
    #    else:
    #       full_desc = self.build_description(action_name)

    #       # Save the full description now
    #       self.general["desc"] = full_desc

    #    # Create the results dir
    #    results_dir = self.results_dir

    #    # Write out the config
    #    self.save(os.path.join(results_dir, "config.json"))

    # @property
    # def results_dir(self):
    #    if self._results_dir is None:
    #       self._results_dir = misc.create_result_subdir(self.general["logs_dir"], self.general["run_idx"], self.general["desc"])

    #    return self._results_dir

    # def build_description(self, action: str) -> str:

    #    ds_str = self._state["dataset"]["name"]

    #    desc_str = self._state["general"]["desc"]

    #    gpu_count = int(self._state["general"]["gpu_count"])
    #    gpu_str = "1gpu" if gpu_count == 1 else f"{gpu_count}gpus"

    #    # Start with the action
    #    desc = f"{action}-{ds_str}-{desc_str}-{gpu_str}"

    #    return desc

    # def _complete_schedule(self):

    #    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    #    schedules: Dict[str, Dict[int, T]] = self.schedule

    #    for key, sched in schedules._state.items():

    #       curr_value = sched[4]

    #       for res in sorted(resolutions):

    #          curr_value = sched.get(res, curr_value)
    #          sched[res] = curr_value

    def to_string(self):
        pp = pprint.PrettyPrinter(indent=2, width=80)

        return pp.pformat(dataclasses.asdict(self))
