import asyncio
import concurrent.futures
import dataclasses
import io
import json
import os
import queue
import re
import sys
import threading
import time
import typing
from abc import ABC, abstractmethod
from asyncio.events import get_event_loop
from collections import defaultdict, deque
from ctypes import c_void_p
from functools import reduce

import cudf
import cupy as cp
# import pycuda.autoinit
# import pycuda.driver as cuda
import docker
import grpc.aio
import numpy as np
import pandas as pd
import streamz
import torch
from distributed.client import wait
from streamz import Source
from streamz.core import Stream
from streamz.dataframe import DataFrame
from torch.utils.dlpack import from_dlpack, to_dlpack
from tornado import gen
from tornado.ioloop import IOLoop
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from config import Config
from cudf_subword_helper import Feature, tokenize_text_series
from request import InferenceMemory, Message, MessageMeta, MultiInferenceMessage, MultiMessage, MultiRequest, MultiResponse, MultiResponseMessage, ResponseMemory, SingleRequest, SingleResponse
import async_map  # Do not delete

# Add generated proto output to the path. Stupid. See https://github.com/protocolbuffers/protobuf/issues/1491
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "services")))

from services import request_pb2, request_pb2_grpc

# # Redirect printing to tqdm
orig_out_err = sys.stdout, sys.stderr


def get_time_ms():
    return round(time.time() * 1000)


def df_onread_cleanup(x: typing.Union[cudf.DataFrame, pd.DataFrame]):
    """
    Fixes parsing issues when reading from a file. `\n` gets converted to `\\n` for some reason
    """

    x["data"] = x["data"].str.replace('\\n', '\n', regex=False)

    return x


def json_str_records_to_cudf(x: str):

    cudf_df = cudf.io.read_json(io.StringIO(x), engine="cudf", lines=True)

    # Cleanup the \\n -> \n issue
    return df_onread_cleanup(cudf_df)


config = Config.get()


class StreamWrapper(ABC):
    def __init__(self, c: Config):
        # self._prev_stage: Stage = None
        self._input_stream: Stream = None
        self._output_stream: Stream = None
        self._pipeline: Pipeline = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def build(self, input_stream: Stream) -> Stream:
        pass


class SourceStage(StreamWrapper):
    def __init__(self, c: Config):
        super().__init__(c)

        self._post_sink_fn: typing.Callable[[typing.Any], None] = None
        self._done_callbacks: typing.List[typing.Callable] = []

    @property
    def input_count(self) -> int:
        # Return None for no max intput count
        return None

    def add_done_callback(self, cb):
        self._done_callbacks.append(cb)

    async def build(self, input_stream: Stream) -> Stream:

        assert input_stream is None, "Sources shouldnt have input streams"

        self._input_stream = None

        self._output_stream = await self._build()

        if (self._post_sink_fn is not None):
            self._output_stream.sink(self._post_sink_fn)

        return self._output_stream

    @abstractmethod
    async def _build(self) -> Stream:

        pass


class FileSourceStage(SourceStage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._filename = ".tmp/dataset4/pcap_dump_augmented_0delay.json"
        self._input_count = None

    @property
    def name(self) -> str:
        return "File Source"

    @property
    def input_count(self) -> int:
        # Return None for no max intput count
        return self._input_count

    async def _build(self) -> Stream:
        def json_to_cudf(in_str: typing.List[str]):

            # This method is slow but it works well
            # df = cudf.from_pandas(pd.DataFrame.from_records([json.loads(x) for x in in_str]))

            df = json_str_records_to_cudf("".join(in_str))
            return df

        # Read the number of lines for progress reporting
        # TODO: Make this async
        self._input_count = sum(1 for i in open(self._filename, 'rb'))

        source: Source = Stream.from_textfile(self._filename, asynchronous=True, loop=IOLoop.current()).rate_limit(
            1 / 10000).timed_window(0.1).filter(lambda x: len(x) > 0)

        source = source.async_map(json_to_cudf, executor=self._pipeline.thread_pool)

        return source


class FileSourceStage2(SourceStage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._filename = ".tmp/dataset4/pcap_dump_augmented_0delay.json"
        self._input_count = None
        

    @property
    def name(self) -> str:
        return "File Source"

    @property
    def input_count(self) -> int:
        # Return None for no max intput count
        return self._input_count

    async def _build(self) -> Stream:

        df = cudf.read_json(self._filename, engine="cudf", lines=True)

        df = df_onread_cleanup(df)

        source: Source = Stream.from_iterable(self._generate_frames(df), asynchronous=True, loop=IOLoop.current())

        def fix_df(x: cudf.DataFrame):
            # Reset the index so they all get a unique index ID
            return x[1].reset_index(drop=True)

        # source = source.map(fix_df)

        return source

    def _generate_frames(self, df):
        for x in df.groupby(np.arange(len(df)) // 256):
            y = x[1].reset_index(drop=True)

            yield y

        for cb in self._done_callbacks:
            cb()


class KafkaSourceStage(SourceStage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._consumer_conf = {
            'bootstrap.servers': c.kafka.bootstrap_servers, 'group.id': 'custreamz', 'session.timeout.ms': "60000"
        }

        self._input_topic = c.kafka.input_topic
        self._use_dask = False

    @property
    def name(self) -> str:
        return "Kafka Source"

    async def _build(self) -> Stream:

        if (self._use_dask):
            from dask.distributed import Client
            client = Client()

            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=True,
                                                       engine="cudf",
                                                       poll_interval="10millis",
                                                       loop=IOLoop.current(),
                                                       max_batch_size=1000)
        else:
            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=False,
                                                       engine="cudf",
                                                       poll_interval="10millis",
                                                       loop=IOLoop.current(),
                                                       max_batch_size=1000)

        # Always gather here (no-op if not using dask)
        return source.gather()


class Stage(StreamWrapper):
    def __init__(self, c: Config):
        super().__init__(c)

        self._pre_sink_fn: typing.Callable[[typing.Any], None] = None
        self._post_sink_fn: typing.Callable[[typing.Any], None] = None

        self._timestamps: typing.Dict[int, int] = {}

    async def build(self, input_stream: Stream) -> Stream:

        assert input_stream is not None, "Sources must have input streams"

        self._input_stream = input_stream

        if (self._pre_sink_fn is not None):
            self._input_stream.sink(self._pre_sink_fn)

        self._output_stream = await self._build(input_stream)

        if (self._post_sink_fn is not None):
            self._output_stream.sink(self._post_sink_fn)

        return self._output_stream

    async def _build(self, input_stream: Stream) -> Stream:

        pass


class BufferStage(Stage):
    def __init__(self, c: Config, buffer_count: int = 1000):
        super().__init__(c)

        self._buffer_count = buffer_count

    @property
    def name(self) -> str:
        return "buffer"

    async def _build(self, input_stream: Stream) -> Stream:
        return input_stream.buffer(self._buffer_count)


class TqdmStage(Stage):
    def __init__(self,
                 c: Config,
                 progress_desc: str = "Progress",
                 smoothing: int = 0.05,
                 unit="messages",
                 determine_count_fn: typing.Callable[[typing.Any], int] = None):
        super().__init__(c)

        self._progress = tqdm(desc=progress_desc,
                              smoothing=smoothing,
                              dynamic_ncols=True,
                              unit=unit,
                              mininterval=1.0,
                              maxinterval=2.0)

        # self._pre_sink_fn = self.pre_timestamps
        self._post_sink_fn = self._progress_sink

        self._determine_count_fn = determine_count_fn

    @property
    def name(self) -> str:
        return "progress"

    async def _build(self, input_stream: Stream) -> Stream:

        self._pipeline._source_stage.add_done_callback(self._refresh_progress)

        return input_stream

    def _refresh_progress(self):
        self._progress.refresh()

    def _progress_sink(self, x):

        if (self._determine_count_fn is None):
            self._determine_count_fn = self._auto_count_fn(x)

        # Skip incase we have empty objects
        if (self._determine_count_fn is None):
            return

        # This starts the timer on the first message. Otherwise it sits from creation with no updates
        if (self._progress.n == 0):
            self._progress.unpause()

        # Do our best to determine the count
        n = self._determine_count_fn(x)

        self._progress.update(n=n)

    def _auto_count_fn(self, x):

        if (x is None):
            return None

        # Wait for a list thats not empty
        if (isinstance(x, list) and len(x) == 0):
            return None

        if (isinstance(x, cudf.DataFrame)):
            return lambda y: len(y.index)
        elif (isinstance(x, MultiMessage)):
            return lambda y: y.mess_count
        elif (isinstance(x, list)):
            item_count_fn = self._auto_count_fn(x[0])
            return lambda y: reduce(item_count_fn, y, 0)
        elif (isinstance(x, str)):
            return lambda y: 1
        elif (hasattr(x, "__len__")):
            return lambda y: len(y)
        else:
            raise NotImplementedError("Unsupported type: {}".format(type(x)))


class DeserializeStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._lock = threading.Lock()
        self._id_counter = 0

        # self._pre_sink_fn = self.pre_timestamps
        self._post_sink_fn = self.post_timestamps

    @property
    def name(self) -> str:
        return "deserialize"

    def pre_timestamps(self, x: cudf.DataFrame):

        curr_time = get_time_ms()

        # Get IDs
        ids = x["ID"].to_arrow().to_pylist()

        self._timestamps = self._timestamps.fromkeys(ids, curr_time)

    def process_dataframe(self, x: cudf.DataFrame):

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

    async def _build(self, input_stream: Stream) -> Stream:

        return input_stream.async_map(self.process_dataframe, executor=self._pipeline.thread_pool)

    def post_timestamps(self, x: MultiMessage):

        curr_time = get_time_ms()

        # # Get IDs
        # ids = x.id_list

        # deltas = [curr_time - self._timestamps.pop(i) for i in ids]

        x.set_meta("ts_" + self.name, curr_time)


class PreprocessStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._post_sink_fn = self.post_timestamps
        self._seq_length = c.model.seq_length
        self._vocab_hash_file = c.model.vocab_hash_file

    @property
    def name(self) -> str:
        return "preprocess"

    def pre_process_batch(self, x: MultiMessage):

        # Set the stride to 75%. Works well with powers of 2
        stride = self._seq_length // 2
        stride = stride + stride // 2

        tokenized = tokenize_text_series(cudf.Series(x.data_col), self._seq_length, stride, self._vocab_hash_file)

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

    async def _build(self, input_stream: Stream) -> Stream:

        stream = input_stream.async_map(self.pre_process_batch, executor=self._pipeline.thread_pool)

        return stream

    def post_timestamps(self, x: MultiInferenceMessage):

        curr_time = get_time_ms()

        # # Get IDs
        # ids = x.id_list

        # deltas = [curr_time - self._timestamps.pop(i) for i in ids]

        x.set_meta("ts_" + self.name, curr_time)


class InferenceStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._post_sink_fn = self.post_timestamps
        self._seq_length = c.model.seq_length

        self._thread = None
        self._inf_queue = inf_queue = queue.Queue()

        self._max_batch_size = c.model.max_batch_size

        self._output_stream: Stream = Stream(loop=IOLoop.current(), asynchronous=True, ensure_io_loop=True)

    @property
    def name(self) -> str:
        return "inference"

    @abstractmethod
    def _get_inference_fn(self) -> typing.Callable:
        pass

    async def _build(self, input_stream: Stream) -> Stream:

        ready_event = asyncio.Event()

        threading.Thread(target=self._get_inference_fn(), daemon=True, args=(
            IOLoop.current(),
            self._inf_queue,
            ready_event,
        )).start()

        # Wait for the inference thread to be ready
        await ready_event.wait()

        stream = input_stream

        # First convert to manageable batches. If the batches are too large, we cant process them
        stream = stream.async_map(self._split_batches, executor=self._pipeline.thread_pool)

        # Queue the inference work
        stream = stream.async_map(self._queue_inf_work)

        stream = stream.async_map(self._convert_response, executor=self._pipeline.thread_pool)

        return stream

    def _split_batches(self, x: MultiInferenceMessage):

        out_batches = []

        start_idx = 0
        curr_idx = 0
        curr_batch_count = 0

        id_array = cp.concatenate([cp.array([-1]), x.seq_ids[:, 0], cp.array([-1])])

        diff_ids = cp.where(id_array[1:] != id_array[:-1])[0]

        # id_counts = (diff_ids[1:] - diff_ids[:-1]).tolist()

        # # Loop over each incoming message and split
        # for count in id_counts:
        #     if (curr_batch_count + count > self._max_batch_size):
        #         assert curr_batch_count > 0, "Otherwise we need to split even a single id into multiple"

        #         # Build current
        #         assert start_idx != curr_idx, "Cant have 0 elements"

        #         out_batches.append((start_idx, curr_idx))

        #         start_idx = curr_idx + 1

        #     curr_idx = curr_idx + count

        diff_ids = diff_ids.tolist()

        head = 0
        tail = 0

        for i in range(1, len(diff_ids)):

            head_idx = diff_ids[head]
            tail_idx = diff_ids[i] - 1

            poss_count = diff_ids[i] - diff_ids[head]

            if (poss_count > self._max_batch_size):
                out_batches.append((diff_ids[head], diff_ids[tail]))

                head = tail

            tail = i

        out_batches.append((diff_ids[head], diff_ids[tail]))

        out_resp = []

        for start, stop in out_batches:

            out_resp.append(x.get_slice(start, stop))

        assert len(out_resp) > 0

        return out_resp

    async def _queue_inf_work(self, x: typing.List[MultiInferenceMessage]):
        # Get the current event loop.
        loop = asyncio.get_running_loop()

        futures = []

        for y in x:
            fut = loop.create_future()

            self._inf_queue.put((y, fut))

            futures.append(fut)

        res = await asyncio.gather(*futures, return_exceptions=True)

        return x, res

    def _convert_response(self, x: typing.Tuple[typing.List[MultiInferenceMessage], typing.List[ResponseMemory]]):

        # Convert a MultiResponse into a MultiResponseMessage
        in_message = x[0]
        out_message = x[1]

        assert len(in_message) == len(out_message)

        # Get the total output size
        total_mess_count = reduce(lambda y, z: y + z.mess_count, in_message, 0)

        # Create a message data to store the entire list
        memory = ResponseMemory(count=total_mess_count, probs=cp.zeros((total_mess_count, out_message[0].probs.shape[1])))

        saved_meta = in_message[0].meta
        saved_offset = in_message[0].mess_offset
        saved_count = 0

        for inf, res in zip(in_message, out_message):

            # Ensure they all share the same meta object. Otherwise this doesnt work
            assert inf.meta == saved_meta

            # Make sure we have a continuous list
            assert inf.mess_offset == saved_offset + saved_count

            # Two scenarios:
            if (inf.mess_count == inf.count):
                # In message and out message have same count. Just use probs as is
                memory.probs[inf.mess_offset:inf.mess_offset + inf.mess_count, :] = res.probs
            else:
                assert inf.count == res.count

                mess_ids = inf.seq_ids[:, 0].get().tolist()

                # Out message has more reponses, so we have to do key based blending of probs
                for i, id in enumerate(mess_ids):
                    memory.probs[id, :] = cp.maximum(memory.probs[id, :], res.probs[i, :])

            saved_count += inf.mess_count

        # For now, we assume that this is the whole group of messages so it must start at 0
        assert saved_offset == 0

        return MultiResponseMessage(meta=saved_meta,
                                    mess_offset=saved_offset,
                                    mess_count=saved_count,
                                    memory=memory,
                                    offset=0,
                                    count=memory.count)

    def post_timestamps(self, x: MultiResponseMessage):

        curr_time = get_time_ms()

        # # Get IDs
        # ids = x.id_list

        # deltas = [curr_time - self._timestamps.pop(i) for i in ids]

        x.set_meta("ts_" + self.name, curr_time)


class WriteClassificationStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._seq_length = c.model.seq_length
        self._vocab_hash_file = c.model.vocab_hash_file

    @property
    def name(self) -> str:
        return "write_classification"

    def pre_process_batch(self, x: MultiMessage):

        # Set the stride to 75%. Works well with powers of 2
        stride = self._seq_length // 2
        stride = stride + stride // 2

        tokenized = tokenize_text_series(cudf.Series(x.data_col), self._seq_length, stride, self._vocab_hash_file)

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

    async def _build(self, input_stream: Stream) -> Stream:

        stream = input_stream.async_map(self.pre_process_batch, executor=self._pipeline.thread_pool)

        return stream

    def post_timestamps(self, x: MultiInferenceMessage):

        curr_time = get_time_ms()

        # # Get IDs
        # ids = x.id_list

        # deltas = [curr_time - self._timestamps.pop(i) for i in ids]

        x.set_meta("ts_" + self.name, curr_time)

class Pipeline():
    def __init__(self, c: Config):

        self._inf_queue = queue.Queue()

        self._source_stage: SourceStage = None
        self._source_count: int = None  # Maximum number of iterations for progress reporting. None = Unknown/Unlimited

        self._id_counter = 0

        self._stages: typing.List[Stage] = []

        self._source_stream: Source = None

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @property
    def thread_pool(self):
        return self._thread_pool

    def _add_id_col(self, x: cudf.DataFrame):

        # Data in stream is cudf Dataframes at this point. We need an ID column before continuing
        x.insert(0, 'ID', range(self._id_counter, self._id_counter + len(x)))
        self._id_counter += len(x)

        return x

    def set_source(self, source: SourceStage):

        self._source_stage = source
        source._pipeline = self

    def add_stage(self, stage: Stage):

        self._stages.append(stage)
        stage._pipeline = self

    async def build_and_start(self):

        self._source_stream = await self._source_stage.build(None)

        self._source_stage.add_done_callback(self._on_input_complete)

        # Add the ID single threaded
        current_stream = self._source_stream.map(self._add_id_col)

        # Now loop over stages
        for s in self._stages:
            current_stream = await s.build(current_stream)

        self._source_stream.start()

    def _on_input_complete(self):
        tqdm.write("All Input Complete")

    def run(self):

        loop = asyncio.get_event_loop()

        from grpc_preprocessing import serve

        asyncio.ensure_future(serve())

        asyncio.ensure_future(self.build_and_start())
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
            print("Exited")
