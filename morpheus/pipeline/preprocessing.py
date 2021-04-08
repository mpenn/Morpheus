from functools import reduce
from logging import StreamHandler
from morpheus.pipeline.pipeline import StreamFuture, StreamPair, get_time_ms
import time
import typing
from streamz.core import Stream
from tornado.ioloop import IOLoop
from morpheus.pipeline import Stage
from morpheus.pipeline.messages import InferenceMemory, MessageMeta, MultiInferenceMessage, MultiMessage, MultiResponseMessage
from morpheus.config import Config
from tqdm import tqdm
import cudf
import threading
import json
from morpheus.utils.cudf_subword_helper import tokenize_text_series
import cupy as cp
import typing_utils

class DeserializeStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._use_dask = c.use_dask

        self._post_sink_fn = self.post_timestamps

    @property
    def name(self) -> str:
        return "deserialize"

    def accepted_types(self) -> typing.Tuple:
        return (cudf.DataFrame, StreamFuture[cudf.DataFrame])

    @staticmethod
    def process_dataframe(x: cudf.DataFrame):

        # Convert here to pandas since this will persist after the message is done
        x_pd = x.to_pandas()

        # Now determine the list of input strings before modification
        input_json = [json.dumps(y) for y in x_pd.loc[:, x_pd.columns != 'ID'].to_dict(orient="records")]

        # Add the start_time field
        x_pd["ts_start"] = round(time.time() * 1000)

        # Try to double deserialize
        def deserialize_data(y: str):
            try:
                return str(json.loads(y))
            except:
                return y

        x_pd["data"] = x_pd["data"].apply(deserialize_data)

        # Build the message data
        meta = MessageMeta(df=x_pd, input_json=input_json)

        return MultiMessage(meta=meta, mess_offset=0, mess_count=len(x_pd))

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiMessage

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):

            stream = stream.map(DeserializeStage.process_dataframe)
            out_type = StreamFuture[MultiMessage]
        else:
            stream = stream.async_map(DeserializeStage.process_dataframe, executor=self._pipeline.thread_pool)

        return stream, out_type

    async def post_timestamps(self, x: MultiMessage):

        curr_time = get_time_ms()

        x.set_meta("ts_" + self.name, curr_time)


class PreprocessStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._post_sink_fn = self.post_timestamps
        self._seq_length = c.model_seq_length
        self._vocab_hash_file = c.model_vocab_hash_file

    @property
    def name(self) -> str:
        return "preprocess"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, StreamFuture[MultiMessage])

    @staticmethod
    def pre_process_batch(x: MultiMessage, seq_len: int, stride: int, vocab_hash_file: str):

        tokenized = tokenize_text_series(cudf.Series(x.data_col), seq_len, stride, vocab_hash_file)

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemory(count=tokenized.input_ids.shape[0],
                                 input_ids=tokenized.input_ids,
                                 input_mask=tokenized.input_mask,
                                 seq_ids=tokenized.segment_ids)

        infer_message = MultiInferenceMessage(meta=x.meta,
                                              mess_offset=x.mess_offset,
                                              mess_count=x.mess_count,
                                              memory=memory,
                                              offset=0,
                                              count=memory.count)

        return infer_message

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        # Set the stride to 75%. Works well with powers of 2
        stride = self._seq_length // 2
        stride = stride + stride // 2

        stream = input_stream[0]
        out_type = MultiInferenceMessage

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):
            stream = stream.map(PreprocessStage.pre_process_batch, stride=stride, seq_len=self._seq_length, vocab_hash_file=self._vocab_hash_file)
            out_type = StreamFuture[MultiInferenceMessage]
        else:
            stream = stream.async_map(PreprocessStage.pre_process_batch, executor=self._pipeline.thread_pool, stride=stride, seq_len=self._seq_length, vocab_hash_file=self._vocab_hash_file)

        return stream, out_type

    def post_timestamps(self, x: MultiInferenceMessage):

        curr_time = get_time_ms()

        x.set_meta("ts_" + self.name, curr_time)