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

    @property
    def data_col(self):
        return self.get_meta("data")

    @property
    def data(self) -> typing.List[str]:
        return self.get_meta_list("data")

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
    input_ids: cp.ndarray
    input_mask: cp.ndarray
    seq_ids: cp.ndarray


@dataclasses.dataclass
class MultiInferenceMessage(MultiMessage):

    memory: InferenceMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def input_ids(self):
        return self.memory.input_ids[self.offset:self.offset + self.count, :]

    @property
    def input_mask(self):
        return self.memory.input_mask[self.offset:self.offset + self.count, :]

    @property
    def seq_ids(self):
        return self.memory.seq_ids[self.offset:self.offset + self.count, :]

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
