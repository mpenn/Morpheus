import dataclasses
import typing

import cupy as cp
import pandas as pd


@dataclasses.dataclass
class MessageMeta:
    df: pd.DataFrame
    input_json: typing.List[str]

    @property
    def count(self) -> int:
        return len(self.df)


@dataclasses.dataclass
class Message:
    meta: MessageMeta = dataclasses.field(repr=False)
    meta_idx: int


@dataclasses.dataclass
class MultiMessage:

    meta: MessageMeta = dataclasses.field(repr=False)
    mess_offset: int
    mess_count: int

    @property
    def input_json(self):
        return self.meta.input_json[self.mess_offset:self.mess_offset + self.mess_count]

    # @property
    # def data_col(self):
    #     return self.get_meta("data")

    # @property
    # def data(self) -> typing.List[str]:
    #     return self.get_meta_list("data")

    @property
    def id_col(self):
        return self.get_meta("ID")

    @property
    def id(self) -> typing.List[int]:
        return self.get_meta_list("ID")

    @property
    def timestamp(self) -> typing.List[int]:
        return self.get_meta_list("timestamp")

    def get_meta(self, col_name: str):
        return self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], col_name]

    def get_meta_list(self, col_name: str = None):
        return self.get_meta(col_name=col_name).to_list()

    def set_meta(self, col_name: str, value):
        self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], col_name] = value


@dataclasses.dataclass
class InferenceMemory:
    count: int

    inputs: typing.Dict[str, cp.ndarray] = dataclasses.field(default_factory=dict, init=False)

    def __getattr__(self, name: str) -> typing.Any:

        input_val = self.inputs.get(name, default=None)

        if (input_val is not None):
            return input_val

        return super().__getattr__(name)

    def __setattr__(self, name: str, value: typing.Any) -> None:

        # If its a cupy array, set it to the inputs field
        if (isinstance(value, cp.ndarray)):
            self.inputs[name] = value
            return

        return super().__setattr__(name, value)


@dataclasses.dataclass
class InferenceMemoryNLP(InferenceMemory):

    input_ids: dataclasses.InitVar[cp.ndarray]
    input_mask: dataclasses.InitVar[cp.ndarray]
    seq_ids: dataclasses.InitVar[cp.ndarray]

    def __post_init__(self, input_ids, input_mask, seq_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seq_ids = seq_ids


@dataclasses.dataclass
class InferenceMemoryFIL(InferenceMemory):

    input__0: dataclasses.InitVar[cp.ndarray]
    seq_ids: dataclasses.InitVar[cp.ndarray]

    def __post_init__(self, input__0, seq_ids):
        self.input__0 = input__0
        self.seq_ids = seq_ids


@dataclasses.dataclass
class MultiInferenceMessage(MultiMessage):

    memory: InferenceMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def inputs(self):
        return {key: self.get_input(key) for key in self.memory.inputs.keys()}

    def __getattr__(self, name: str) -> typing.Any:

        input_val = self.memory.inputs.get(name, None)

        if (input_val is not None):
            return input_val[self.offset:self.offset + self.count, :]

        return super().__getattr__(name)

    def get_input(self, name: str):
        return self.memory.inputs[name][self.offset:self.offset + self.count, :]

    def get_slice(self, start, stop):
        mess_start = self.seq_ids[start, 0].item()
        mess_stop = self.seq_ids[stop - 1, 0].item() + 1
        return MultiInferenceMessage(meta=self.meta,
                                     mess_offset=mess_start,
                                     mess_count=mess_stop - mess_start,
                                     memory=self.memory,
                                     offset=start,
                                     count=stop - start)


@dataclasses.dataclass
class MultiInferenceNLPMessage(MultiInferenceMessage):
    @property
    def input_ids(self):
        return self.get_input("input_ids")

    @property
    def input_mask(self):
        return self.get_input("input_mask")

    @property
    def seq_ids(self):
        return self.get_input("seq_ids")


@dataclasses.dataclass
class MultiInferenceFILMessage(MultiInferenceMessage):
    @property
    def input__0(self):
        return self.get_input("input__0")

    @property
    def seq_ids(self):
        return self.get_input("seq_ids")


@dataclasses.dataclass
class ResponseMemory:
    count: int
    probs: cp.ndarray


@dataclasses.dataclass
class MultiResponseMessage(MultiMessage):

    memory: ResponseMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def probs(self) -> cp.ndarray:
        return self.memory.probs[self.offset:self.offset + self.count, :]
