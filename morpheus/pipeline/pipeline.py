import asyncio
import concurrent.futures
from morpheus.pipeline.messages import MultiMessage
import queue
import sys
import time
import typing
from abc import ABC
from abc import abstractmethod

import cudf
import distributed
import typing_utils
from distributed.client import wait
from streamz import Source
from streamz.core import Stream
from tornado.ioloop import IOLoop
from tqdm import tqdm

from morpheus.config import Config

config = Config.get()


def get_time_ms():
    return round(time.time() * 1000)


T = typing.TypeVar('T')

StreamFuture = typing._GenericAlias(distributed.client.Future, T, special=True, inst=False, name="StreamFuture")

StreamPair = typing.Tuple[Stream, typing.Type]


class StreamWrapper(ABC):
    def __init__(self, c: Config):
        self._input_stream: Stream = None
        self._output_stream: Stream = None
        self._pipeline: Pipeline = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def build(self, input_stream: StreamPair) -> StreamPair:
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

    async def build(self, input_stream: StreamPair) -> StreamPair:

        assert input_stream is None, "Sources shouldnt have input streams"

        self._input_stream = None

        output = await self._build()

        self._output_stream = output[0]

        if (self._post_sink_fn is not None):
            self._output_stream.gather().sink(self._post_sink_fn)

        return output

    @abstractmethod
    async def _build(self) -> StreamPair:

        pass


class Stage(StreamWrapper):
    def __init__(self, c: Config):
        super().__init__(c)

        self._pre_sink_fn: typing.Callable[[typing.Any], None] = None
        self._post_sink_fn: typing.Callable[[typing.Any], None] = None

        # Derived classes should set this to true to log timestamps in debug builds
        self._should_log_timestamps = False

    @abstractmethod
    def accepted_types(self) -> typing.Tuple:
        pass

    async def build(self, input_stream: StreamPair) -> StreamPair:

        assert input_stream is not None, "Sources must have input streams"

        # Check the type. Convert Any to object
        if (not typing_utils.issubtype(input_stream[1], typing.Union[self.accepted_types()])):
            raise RuntimeError("The {} stage cannot handle input of {}. Accepted input types: {}".format(
                self.name, input_stream[1], self.accepted_types()))

        self._input_stream = input_stream[0]

        if (self._pre_sink_fn is not None):
            self._input_stream.sink(self._pre_sink_fn)

        output = await self._build(input_stream)

        self._output_stream = output[0]

        if (self._post_sink_fn is not None):
            self._output_stream.gather().sink(self._post_sink_fn)

        if (Config.get().debug and self._should_log_timestamps):
            # Cache the name property. Removes dependency on self in callback
            cached_name = self.name

            def post_timestamps(self, x: MultiMessage):

                curr_time = get_time_ms()

                x.set_meta("ts_" + cached_name, curr_time)

            self._output_stream.gather().sink(post_timestamps)

        # self._output_stream.add_done_callback(self._on_complete)

        return output

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        pass

    def on_start(self):
        pass

    def _on_complete(self, stream: Stream):

        tqdm.write("Stage Complete: {}".format(self.name))


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

        self._use_dask = c.use_dask

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

        if (self._use_dask):
            tqdm.write("====Launching Dask====")
            from distributed import Client
            self._client: Client = await Client(loop=IOLoop.current(), processes=True, asynchronous=True)

        tqdm.write("====Building Pipeline====")

        source_stream_pair = await self._source_stage.build(None)
        self._source_stream = source_stream_pair[0]

        self._source_stage.add_done_callback(self._on_input_complete)

        source_stream_pair[0].sink(self._on_start)

        # Add the ID single threaded
        current_stream_and_type = source_stream_pair
        # current_stream_and_type = current_stream_and_type[0].map(self._add_id_col), current_stream_and_type[1]

        tqdm.write("Added source: {} -> {}".format(self._source_stage.name, str(current_stream_and_type[1])))

        # If using dask, scatter here
        if (self._use_dask):
            if (typing_utils.issubtype(current_stream_and_type[1], typing.List)):
                current_stream_and_type = (current_stream_and_type[0].scatter_batch().flatten(),
                                           StreamFuture[typing.get_args(current_stream_and_type[1])[0]])
            else:
                current_stream_and_type = (current_stream_and_type[0].scatter(), StreamFuture[current_stream_and_type[1]])
        else:
            if (typing_utils.issubtype(current_stream_and_type[1], typing.List)):
                current_stream_and_type = current_stream_and_type[0].flatten(), typing.get_args(current_stream_and_type[1])[0]

        # Now loop over stages
        for s in self._stages:
            current_stream_and_type = await s.build(current_stream_and_type)

            tqdm.write("Added stage: {} -> {}".format(s.name, str(current_stream_and_type[1])))

        tqdm.write("====Starting Inference====")

        self._source_stream.start()

    def _on_start(self, _):

        tqdm.write("Starting! Time: {}".format(time.time()))

        # Loop over all stages and call on_start if it exists
        for s in self._stages:
            s.on_start()

    def _on_input_complete(self):
        tqdm.write("All Input Complete")

    def run(self):

        loop = asyncio.get_event_loop()

        asyncio.ensure_future(self.build_and_start())
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
            print("Exited")
