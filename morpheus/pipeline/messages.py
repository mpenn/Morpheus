import dataclasses
from morpheus.utils.cudf_subword_helper import Feature
import threading
import cupy as cp
import typing
import cudf
import pandas as pd


@dataclasses.dataclass
class RequestData:
    count: int
    input_ids: cp.ndarray
    input_mask: cp.ndarray
    segment_ids: cp.ndarray
    input_str: typing.List[str]
    timestamp: typing.List[int]


@dataclasses.dataclass
class SingleRequest:
    offset: int
    data: RequestData

    @property
    def input_ids(self):
        return self.data.input_ids[self.offset:self.offset + 1, :]

    @property
    def input_mask(self):
        return self.data.input_mask[self.offset:self.offset + 1, :]

    @property
    def segment_ids(self):
        return self.data.segment_ids[self.offset:self.offset + 1, :]

    @property
    def input_str(self):
        return self.data.input_str[self.offset]

    @property
    def timestamp(self):
        return self.data.timestamp[self.offset]


@dataclasses.dataclass
class MultiRequest:
    offset: int
    count: int
    data: RequestData

    @property
    def input_ids(self):
        return self.data.input_ids[self.offset:self.offset + self.count, :]

    @property
    def input_mask(self):
        return self.data.input_mask[self.offset:self.offset + self.count, :]

    @property
    def segment_ids(self):
        return self.data.segment_ids[self.offset:self.offset + self.count, :]

    @property
    def input_str(self):
        return self.data.input_str[self.offset:self.offset + self.count]

    @property
    def timestamp(self):
        return self.data.timestamp[self.offset:self.offset + self.count]

    def to_singles(self):
        out = []

        for i in range(self.count):
            out.append(SingleRequest(data=self.data, offset=i))

        return out

    def create(input_ids: cp.ndarray,
               input_mask: cp.ndarray,
               segment_ids: cp.ndarray,
               input_str: typing.List[str],
               timestamp: typing.List[int]) -> "MultiRequest":
        data = RequestData(
            count=input_ids.shape[0],
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_str=input_str,
            timestamp=timestamp,
        )

        out = MultiRequest(offset=0, count=data.count, data=data)

        return out

    def from_feature(in_feature: Feature, str_series: cudf.Series, time_series: cudf.Series) -> "MultiRequest":
        in_str = str_series.to_arrow().to_pylist() if isinstance(str_series,
                                                                 cudf.Series) else str_series.to_pandas().to_dict('records')
        in_time = time_series.to_arrow().to_pylist() if isinstance(time_series,
                                                                   cudf.Series) else time_series.to_pandas().to_dict('records')

        data = RequestData(count=in_feature.input_ids.shape[0],
                           input_ids=in_feature.input_ids,
                           input_mask=in_feature.input_mask,
                           segment_ids=in_feature.segment_ids,
                           input_str=in_str,
                           timestamp=in_time)

        return MultiRequest(offset=0, count=data.count, data=data)

    def from_singles(single_requests: typing.List[SingleRequest]):

        final_count = len(single_requests)

        # Quick exit if no copying needs to be done
        if (final_count == 1):
            return MultiRequest(offset=single_requests[0].offset, count=final_count, data=single_requests[0].data)

        data = RequestData(
            count=final_count,
            input_ids=cp.empty_like(single_requests[0].input_ids,
                                    shape=(final_count, single_requests[0].input_ids.shape[1]),
                                    order="C"),
            input_mask=cp.empty_like(single_requests[0].input_mask,
                                     shape=(final_count, single_requests[0].input_mask.shape[1]),
                                     order="C"),
            segment_ids=cp.empty_like(single_requests[0].segment_ids,
                                      shape=(final_count, single_requests[0].segment_ids.shape[1]),
                                      order="C"),
            input_str=[""] * final_count,
            timestamp=[0] * final_count,
        )

        for i, r in enumerate(single_requests):
            data.input_ids[i:i + 1, :] = r.input_ids
            data.input_mask[i:i + 1, :] = r.input_mask
            data.segment_ids[i:i + 1, :] = r.segment_ids
            data.input_str[i] = r.input_str
            data.timestamp[i] = r.timestamp

        return MultiRequest(offset=0, count=final_count, data=data)


@dataclasses.dataclass
class ResponseData:
    count: int
    probs: cp.ndarray
    input_str: typing.List[str]
    timestamp: typing.List[int]


@dataclasses.dataclass
class MultiResponse:
    offset: int
    count: int
    data: RequestData

    @property
    def probs(self):
        return self.data.probs[self.offset:self.offset + self.count, :]

    @property
    def input_str(self):
        return self.data.input_str[self.offset:self.offset + self.count]

    @property
    def timestamp(self):
        return self.data.timestamp[self.offset:self.offset + self.count]

    def from_singles(singles: typing.List["SingleResponse"]):

        final_count = len(singles)

        # Quick exit if no copying needs to be done
        if (final_count == 1):
            return MultiResponse(offset=singles[0].offset, count=final_count, data=singles[0].data)

        data = ResponseData(
            count=final_count,
            probs=cp.empty_like(singles[0].probs, shape=(final_count, singles[0].probs.shape[1]), order="C"),
            input_str=[""] * final_count,
            timestamp=[0] * final_count,
        )

        for i, r in enumerate(singles):
            data.probs[i:i + 1, :] = r.probs
            data.input_str[i] = r.input_str
            data.timestamp[i] = r.timestamp

        return MultiResponse(offset=0, count=final_count, data=data)


@dataclasses.dataclass
class SingleResponse:
    offset: int
    data: ResponseData

    @property
    def probs(self):
        return self.data.probs[self.offset:self.offset + 1, :]

    @property
    def input_str(self):
        return self.data.input_str[self.offset]

    @property
    def timestamp(self):
        return self.data.timestamp[self.offset]

    def from_multi(multi_res: MultiResponse):
        data = multi_res.data

        out = []

        for i in range(data.count):
            out.append(SingleResponse(data=data, offset=i))

        return out


@dataclasses.dataclass
class MessageMeta:
    df: pd.DataFrame
    input_json: typing.List[str]

    @property
    def count(self) -> int:
        return len(self.df)


message_counter = 0
message_counter_lock = threading.Lock()


def get_next_id():
    with message_counter_lock:
        next_id = message_counter
        message_counter += 1
        return next_id


@dataclasses.dataclass
class Message:
    meta: MessageMeta = dataclasses.field(repr=False)
    meta_idx: int


@dataclasses.dataclass
class MultiMessage:
    # def __init__(self, meta: MessageMeta, count: int, offset: int):

    #     self._meta = meta
    #     self.count = count
    #     self.offset = offset

    meta: MessageMeta = dataclasses.field(repr=False)
    mess_offset: int
    mess_count: int

    # @property
    # def meta(self):
    #     return self._meta

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
        return self.meta.df.loc[self.mess_offset:self.mess_offset + self.mess_count, col_name]

    def get_meta_list(self, col_name: str = None):
        return self.get_meta(col_name=col_name).to_list()

    def set_meta(self, col_name: str, value):
        self.meta.df.loc[self.mess_offset:self.mess_offset + self.mess_count, col_name] = value


@dataclasses.dataclass
class InferenceMemory:
    count: int
    input_ids: cp.ndarray
    input_mask: cp.ndarray
    seq_ids: cp.ndarray


@dataclasses.dataclass
class MultiInferenceMessage(MultiMessage):
    # def __init__(self,
    #              meta: MessageMeta,
    #              count: int,
    #              offset: int,
    #              memory: InferenceMemory):
    #     super().__init__(meta=meta, count=count, offset=offset)

    #     self._memory = memory

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
    # def __init__(self,
    #              meta: MessageMeta,
    #              count: int,
    #              offset: int,
    #              memory: ResponseMemory):
    #     super().__init__(meta=meta, count=count, offset=offset)

    #     self._memory = memory

    memory: ResponseMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def probs(self) -> cp.ndarray:
        return self.memory.probs[self.offset:self.offset + self.count, :]
