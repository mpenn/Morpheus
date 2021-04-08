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
import shutil
import warnings

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


class FileSourceStage2(SourceStage):
    def __init__(self, c: Config, filename: str):
        super().__init__(c)

        self._filename = filename
        self._batch_size = c.pipeline_batch_size
        self._input_count = None

    @property
    def name(self) -> str:
        return "from-file"

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
        for x in df.groupby(np.arange(len(df)) // self._batch_size):
            y = x[1].reset_index(drop=True)

            yield y

        for cb in self._done_callbacks:
            cb()


class KafkaSourceStage(SourceStage):
    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: str = "test_pcap",
                 group_id: str = "custreamz",
                 use_dask: bool = False,
                 poll_interval: str = "10millis"):
        super().__init__(c)

        self._consumer_conf = {'bootstrap.servers': bootstrap_servers, 'group.id': group_id, 'session.timeout.ms': "60000"}

        self._input_topic = input_topic
        self._use_dask = use_dask
        self._poll_interval = poll_interval
        self._max_batch_size = c.pipeline_batch_size

    @property
    def name(self) -> str:
        return "from-kafka"

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
                                                       poll_interval=self._poll_interval,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)
        else:
            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=False,
                                                       engine="cudf",
                                                       poll_interval=self._poll_interval,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)

        # Always gather here (no-op if not using dask)
        return source.gather()


class Stage(StreamWrapper):
    def __init__(self, c: Config):
        super().__init__(c)

        self._pre_sink_fn: typing.Callable[[typing.Any], None] = None
        self._post_sink_fn: typing.Callable[[typing.Any], None] = None

        self._timestamps: typing.Dict[int, int] = {}

    @abstractmethod
    def accepted_types(self) -> typing.Tuple:
        pass

    async def build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        assert input_stream is not None, "Sources must have input streams"
        
        # Check the type. Convert Any to object
        if (not issubclass(input_stream[1], tuple(map(lambda x: object if x is typing.Any else x, self.accepted_types())))):
            raise RuntimeError("The {} stage cannot handle input of {}. Accepted input types: {}".format(
                self.name, input_stream[1], self.accepted_types()))

        self._input_stream = input_stream[0]

        if (self._pre_sink_fn is not None):
            self._input_stream.sink(self._pre_sink_fn)

        output = await self._build(input_stream)

        self._output_stream = output[0]

        if (self._post_sink_fn is not None):
            self._output_stream.sink(self._post_sink_fn)

        return output

    async def _build(self: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        pass


class BufferStage(Stage):
    def __init__(self, c: Config, count: int = 1000):
        super().__init__(c)

        self._buffer_count = count

    @property
    def name(self) -> str:
        return "buffer"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:
        return input_stream[0].buffer(self._buffer_count), input_stream[1]


class TqdmStage(Stage):
    def __init__(self,
                 c: Config,
                 description: str = "Progress",
                 smoothing: int = 0.05,
                 unit="messages",
                 determine_count_fn: typing.Callable[[typing.Any], int] = None):
        super().__init__(c)

        self._progress = tqdm(desc=description,
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
        return "monitor"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

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

    def accepted_types(self) -> typing.Tuple:
        return (cudf.DataFrame, )

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

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        return input_stream[0].async_map(self.process_dataframe, executor=self._pipeline.thread_pool), MultiMessage

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
        self._seq_length = c.model_seq_length
        self._vocab_hash_file = c.model_vocab_hash_file

    @property
    def name(self) -> str:
        return "preprocess"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, )

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

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        stream = input_stream[0].async_map(self.pre_process_batch, executor=self._pipeline.thread_pool)

        return stream, MultiInferenceMessage

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
        self._seq_length = c.model_seq_length

        self._thread = None
        self._inf_queue = queue.Queue()

        self._max_batch_size = c.model_max_batch_size

        # self._progress = tqdm(desc="Inferece",
        #                       smoothing=0.001,
        #                       dynamic_ncols=True,
        #                       unit="inf",
        #                       mininterval=1.0,
        #                       maxinterval=2.0)

    @property
    def name(self) -> str:
        return "inference"

    def accepted_types(self) -> typing.Tuple:
        return (MultiInferenceMessage, )

    @abstractmethod
    def _get_inference_fn(self) -> typing.Callable:
        pass

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        wait_events = []

        for i in range(1):
            ready_event = asyncio.Event()

            threading.Thread(target=self._get_inference_fn(),
                             daemon=True,
                             args=(
                                 IOLoop.current(),
                                 self._inf_queue,
                                 ready_event,
                             )).start()

            # Wait for the inference thread to be ready
            wait_events.append(ready_event.wait())

        asyncio.gather(*wait_events, return_exceptions=True)

        stream = input_stream[0]

        # First convert to manageable batches. If the batches are too large, we cant process them
        stream = stream.async_map(self._split_batches, executor=self._pipeline.thread_pool)

        # Queue the inference work
        stream = stream.async_map(self._queue_inf_work)

        stream = stream.async_map(self._convert_response, executor=self._pipeline.thread_pool)

        return stream, MultiResponseMessage

    def _split_batches(self, x: MultiInferenceMessage):

        out_batches = []

        id_array = cp.concatenate([cp.array([-1]), x.seq_ids[:, 0], cp.array([-1])])

        diff_ids = cp.where(id_array[1:] != id_array[:-1])[0]

        diff_ids = diff_ids.tolist()

        head = 0
        tail = 0

        for i in range(1, len(diff_ids)):

            poss_count = diff_ids[i] - diff_ids[head]

            if (poss_count > self._max_batch_size):
                out_batches.append((diff_ids[head], diff_ids[tail]))

                head = tail

            tail = i

        out_batches.append((diff_ids[head], diff_ids[tail]))

        out_resp = []

        for start, stop in out_batches:

            out_resp.append(x.get_slice(start, stop))

            # self._progress.update(out_resp[-1].mess_count)

        assert len(out_resp) > 0

        return out_resp

    async def _queue_inf_work(self, x: typing.List[MultiInferenceMessage]):
        # Get the current event loop.
        loop = asyncio.get_running_loop()

        futures = []

        for y in x:
            fut = loop.create_future()

            # self._progress.update(n=y.mess_count)

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


class AddClassificationsStage(Stage):
    def __init__(self, c: Config, threshold: float = 0.5):
        super().__init__(c)

        self._threshold = threshold

    @property
    def name(self) -> str:
        return "add-class"

    def accepted_types(self) -> typing.Tuple:
        return (MultiResponseMessage, )

    def _add_labels(self, x: MultiResponseMessage):
        # Keys
        idx2label = {
            0: 'address',
            1: 'bank_acct',
            2: 'credit_card',
            3: 'email',
            4: 'govt_id',
            5: 'name',
            6: 'password',
            7: 'phone_num',
            8: 'secret_keys',
            9: 'user'
        }

        probs_np = (x.probs > self._threshold).astype(cp.bool).get()

        for i, label in idx2label.items():
            x.set_meta("si_" + label, probs_np[:, i].tolist())

        # Return list of strs to write out
        return x

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        stream = input_stream[0]

        # Convert the messages to rows of strings
        stream = stream.async_map(self._add_labels, executor=self._pipeline.thread_pool)

        # Return input unchanged
        return stream, MultiResponseMessage


class WriteToFileStage(Stage):
    def __init__(self, c: Config, output_file: str, overwrite: bool):
        super().__init__(c)

        self._output_file = output_file
        self._overwrite = overwrite
        self._ignore_columns = [r'^ID$', r'^ts_']

        if (os.path.exists(self._output_file)):
            if (self._overwrite):
                os.remove(self._output_file)
            else:
                raise FileExistsError("Cannot output classifications to '{}'. File exists and overwrite = False".format(
                    self._output_file))

    @property
    def name(self) -> str:
        return "to-file"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, )

    def _convert_to_json(self, x: MultiMessage):

        # Get list of columns that pass ignore regex
        columns = list(x.meta.df.columns)

        for test in self._ignore_columns:
            columns = [y for y in columns if not re.match(test, y)]

        # Get metadata from columns
        df = x.get_meta(columns)

        def double_serialize(y: str):
            try:
                return json.dumps(json.dumps(json.loads(y)))
            except:
                return y

        # Special processing for the data column (need to double serialize to match input)
        if ("data" in df):
            df["data"] = df["data"].apply(double_serialize)

        # Convert to list of json string objects
        output_strs = [json.dumps(y) + "\n" for y in df.to_dict(orient="records")]

        # Return list of strs to write out
        return output_strs

    def write_to_file(self, x: typing.List[str]):
        with open(self._output_file, "a") as f:
            f.writelines(x)

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        # Convert the messages to rows of strings
        stream = input_stream[0].async_map(self._convert_to_json, executor=self._pipeline.thread_pool)

        # Sink to file
        stream.sink(self.write_to_file)

        # Return input unchanged
        return input_stream


class FilterDetectionsStage(Stage):
    def __init__(self, c: Config, threshold: float = 0.5):
        super().__init__(c)

        # Probability to consider a detection
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "filter"

    def accepted_types(self) -> typing.Tuple:
        return (MultiResponseMessage, )

    def filter(self, x: MultiResponseMessage) -> typing.List[MultiResponseMessage]:

        # Unfortunately we have to convert this to a list in case there are non-contiguous groups
        output_list = []

        # Get per row detections
        detections = (x.probs > self._threshold).any(axis=1)

        # Surround in False to ensure we get an even number of pairs
        detections = cp.concatenate([cp.array([False]), detections, cp.array([False])])

        true_pairs = cp.where(detections[1:] != detections[:-1])[0].reshape((-1, 2))

        for pair in true_pairs:
            pair = tuple(pair.tolist())
            mess_offset = x.mess_offset + pair[0]
            mess_count = pair[1] - pair[0]

            output_list.append(
                MultiResponseMessage(x.meta,
                                     mess_offset=mess_offset,
                                     mess_count=mess_count,
                                     memory=x.memory,
                                     offset=pair[0],
                                     count=mess_count))

        return output_list

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        stream = input_stream[0]

        # Reduce messages to only have detections
        stream = stream.async_map(self.filter, executor=self._pipeline.thread_pool)

        # Convert list back to single MultiResponseMessage
        stream = stream.flatten()

        # Filter out empty message groups
        stream = stream.filter(lambda x: x.count > 0)

        return stream, MultiResponseMessage


class WriteToKafkaStage(Stage):
    def __init__(self, c: Config, bootstrap_servers: str, output_topic: str):
        super().__init__(c)

        self._kafka_conf = {'bootstrap.servers': bootstrap_servers}

        self._output_topic = output_topic

    @property
    def name(self) -> str:
        return "write_kafka"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        # Write to kafka
        input_stream[0].to_kafka(self._output_topic, self._kafka_conf)

        # Return input unchanged
        return input_stream


class GenerateVizFramesStage(Stage):
    def __init__(self, c: Config, out_dir: str = "./viz_frames", overwrite: bool = False):
        super().__init__(c)

        self._out_dir = out_dir
        self._overwrite = overwrite

        if (os.path.exists(self._out_dir)):
            if (self._overwrite):
                shutil.rmtree(self._out_dir)
            elif (len(list(os.listdir(self._out_dir))) > 0):
                warnings.warn(
                    "Viz output directory '{}' already exists. Errors will occur if frames try to be written over existing files. Suggest emptying the directory or setting `overwrite=True`"
                    .format(self._out_dir))

        os.makedirs(self._out_dir, exist_ok=True)

        self._first_timestamp = -1

    @property
    def name(self) -> str:
        return "gen_viz"

    def accepted_types(self) -> typing.Tuple:
        return (MultiResponseMessage, )

    @staticmethod
    def round_to_sec(x):
        return int(round(x / 1000.0) * 1000)

    def _to_vis_df(self, x: MultiResponseMessage):

        idx2label = {
            0: 'address',
            1: 'bank_acct',
            2: 'credit_card',
            3: 'email',
            4: 'govt_id',
            5: 'name',
            6: 'password',
            7: 'phone_num',
            8: 'secret_keys',
            9: 'user'
        }

        df = x.get_meta(["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "data"])

        def indent_data(y: str):
            try:
                return json.dumps(json.loads(), indent=3)
            except:
                return y

        df["data"] = df["data"].apply(indent_data)

        pass_thresh = (x.probs >= 0.5).any(axis=1)
        max_arg = x.probs.argmax(axis=1)

        condlist = [pass_thresh]

        choicelist = [max_arg]

        index_sens_info = np.select(condlist, choicelist, default=len(idx2label))

        df["si"] = pd.Series(np.choose(index_sens_info.get(), list(idx2label.values()) + ["none"]).tolist())

        df["ts_round_sec"] = (df["timestamp"] / 1000.0).astype(int) * 1000

        # Return a list of tuples of (ts_round_sec, dataframe)
        return [(key, group) for key, group in df.groupby(df.ts_round_sec)]

    def _write_viz_file(self, x: typing.List[typing.Tuple[int, pd.DataFrame]]):

        curr_timestamp = x[0][0]

        in_df = pd.concat([df for _, df in x], ignore_index=True).sort_values(by=["timestamp"])

        # curr_timestamp = GenerateVizFramesStage.round_to_sec(in_df["timestamp"].iloc[0])

        if (self._first_timestamp == -1):
            self._first_timestamp = curr_timestamp

        offset = (curr_timestamp - self._first_timestamp) / 1000

        fn = os.path.join(self._out_dir, "{}.csv".format(offset))

        assert not os.path.exists(fn)

        in_df.to_csv(fn, columns=["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "si", "data"])

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        stream = input_stream[0]

        # Convert stream to dataframes
        stream = stream.map(self._to_vis_df)  # Convert group to dataframe

        # Flatten the list of tuples
        stream = stream.flatten()

        # Partition by group times
        stream = stream.partition(10000, timeout=10, key=lambda x: x[0])  # Group
        # stream = stream.filter(lambda x: len(x) > 0)

        stream.sink(self._write_viz_file)

        # Return input unchanged
        return input_stream


class Pipeline():
    def __init__(self, c: Config):

        self._inf_queue = queue.Queue()

        self._source_stage: SourceStage = None
        self._source_count: int = None  # Maximum number of iterations for progress reporting. None = Unknown/Unlimited

        self._id_counter = 0

        self._stages: typing.List[Stage] = []

        self._source_stream: Source = None

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=c.num_threads)

        self.batch_size = c.pipeline_batch_size

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

        tqdm.write("====Building Pipeline====")

        self._source_stream = await self._source_stage.build(None)

        self._source_stage.add_done_callback(self._on_input_complete)

        # Add the ID single threaded
        current_stream = self._source_stream.map(self._add_id_col)
        currend_stream_and_type = current_stream, cudf.DataFrame

        tqdm.write("Added source: {} -> {}".format(self._source_stage.name, currend_stream_and_type[1].__name__))

        # Now loop over stages
        for s in self._stages:
            currend_stream_and_type = await s.build(currend_stream_and_type)

            tqdm.write("Added stage: {} -> {}".format(s.name, currend_stream_and_type[1].__name__))

        tqdm.write("====Starting Inference====")

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
