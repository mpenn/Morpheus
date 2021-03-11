import dataclasses
import cupy as cp
import typing
from cudf_subword_helper import Feature, tokenize_text_series
import cudf

# @dataclasses.dataclass
# class Request:
#     count: int
#     input_ids: cp.ndarray
#     input_mask: cp.ndarray
#     segment_ids: cp.ndarray
#     input_str: typing.List[str]

#     def split(self, n: int):

#         assert n < self.count

#         # Chop the top off
#         remain_count = self.count - n

#         remain = Request(
#             count=self.count - n,
#             input_ids=self.input_ids[n:, :],
#             input_mask=self.input_mask[n:, :],
#             segment_ids=self.segment_ids[n:, :],
#             input_str=self.input_str[n:],
#         )

#         # reduce
#         self.count = n
#         self.input_ids = self.input_ids[:n, :]
#         self.input_mask = self.input_mask[:n, :]
#         self.segment_ids = self.segment_ids[:n, :]
#         self.input_str = self.input_str[:n]

#         return remain

#     def from_feature(in_feature: Feature, in_df: cudf.Series):
#         return Request(count=in_feature.input_ids.shape[0],
#                        input_ids=in_feature.input_ids,
#                        input_mask=in_feature.input_mask,
#                        segment_ids=in_feature.segment_ids,
#                        input_str=in_df.to_arrow().to_pylist())


@dataclasses.dataclass
class RequestData:
    count: int
    input_ids: cp.ndarray
    input_mask: cp.ndarray
    segment_ids: cp.ndarray
    input_str: typing.List[str]

    def from_feature(in_feature: Feature, in_df: cudf.Series):

        in_str = in_df.to_arrow().to_pylist() if isinstance(in_df, cudf.Series) else in_df.to_pandas().to_dict('records')

        return RequestData(count=in_feature.input_ids.shape[0],
                           input_ids=in_feature.input_ids,
                           input_mask=in_feature.input_mask,
                           segment_ids=in_feature.segment_ids,
                           input_str=in_str)


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

    def from_feature(in_feature: Feature, in_df: cudf.Series) -> typing.List["SingleRequest"]:
        data = RequestData.from_feature(in_feature=in_feature, in_df=in_df)

        out = []

        for i in range(data.count):
            out.append(SingleRequest(data=data, offset=i))

        return out


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

    def to_singles(self):
        out = []

        for i in range(self.count):
            out.append(SingleRequest(data=self.data, offset=i))

        return out

    def create(input_ids: cp.ndarray, input_mask: cp.ndarray, segment_ids: cp.ndarray,
               input_str: typing.List[str]) -> "MultiRequest":
        data = RequestData(count=input_ids.shape[0],
                           input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           input_str=input_str)

        out = MultiRequest(offset=0, count=data.count, data=data)

        return out

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
        )

        for i, r in enumerate(single_requests):
            data.input_ids[i:i + 1, :] = r.input_ids
            data.input_mask[i:i + 1, :] = r.input_mask
            data.segment_ids[i:i + 1, :] = r.segment_ids
            data.input_str[i] = r.input_str

        return MultiRequest(offset=0, count=final_count, data=data)


@dataclasses.dataclass
class ResponseData:
    count: int
    probs: cp.ndarray
    input_str: typing.List[str]


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

    def from_multi(multi_res: MultiResponse):
        data = multi_res.data

        out = []

        for i in range(data.count):
            out.append(SingleResponse(data=data, offset=i))

        return out
