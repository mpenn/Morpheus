import asyncio
import concurrent.futures
import dataclasses
import io
import json
from morpheus.config import Config
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

from morpheus.utils.async_map import async_map

# # Add generated proto output to the path. Stupid. See https://github.com/protocolbuffers/protobuf/issues/1491
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "services")))

# from services import request_pb2, request_pb2_grpc

# # Redirect printing to tqdm
orig_out_err = sys.stdout, sys.stderr


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

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        pass


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

        # from grpc_preprocessing import serve

        # asyncio.ensure_future(serve())

        asyncio.ensure_future(self.build_and_start())
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
            print("Exited")
