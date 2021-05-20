import asyncio
import concurrent.futures
import logging
import queue
import sys
import time
import typing
from abc import ABC, abstractmethod

import cudf
import distributed
import typing_utils
from distributed.client import wait
from morpheus.config import Config
from morpheus.pipeline.messages import MultiMessage
from streamz import Source
from streamz.core import Stream
from tornado.ioloop import IOLoop
from tqdm import tqdm

config = Config.get()

logger = logging.getLogger(__name__)


def get_time_ms():
    return round(time.time() * 1000)


T = typing.TypeVar('T')

StreamFuture = typing._GenericAlias(distributed.client.Future, T, special=True, inst=False, name="StreamFuture")

StreamPair = typing.Tuple[Stream, typing.Type]


class StreamWrapper(ABC):
    """
    This abstract class serves as the morpheus.pipeline's base class. This class wraps a `streamz.Stream`
    object and aids in hooking stages up together. 

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        self._input_stream: Stream = None
        self._output_stream: Stream = None
        self._pipeline: Pipeline = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the stage. Used in logging. Each derived class should override this property with a unique
        name.

        Returns
        -------
        str
            Name of a stage

        """
        pass

    @abstractmethod
    async def build(self, input_stream: StreamPair) -> StreamPair:
        """
        This function is responsible for constructing this Stage's internal `streamz.Stream` object. The input
        of this function is the returned value from the upstream stage.

        The input value is a `StreamPair` which is a tuple containing the input `streamz.Stream` object and
        the message data type.
        
        Parameters
        ----------
        input_stream : StreamPair
            A tuple containing the input `streamz.Stream` object and the message data type.

        Returns
        -------
        StreamPair
            A tuple containing the output `streamz.Stream` object from this stage and the message data type.

        """
        pass


class SourceStage(StreamWrapper):
    """
    The SourceStage is mandatory for the Morpheus pipeline to run. This stage represents the start of the pipeline. All `SourceStage` object take no input but generate output.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._post_sink_fn: typing.Callable[[typing.Any], None] = None
        self._done_callbacks: typing.List[typing.Callable] = []

    @property
    def input_count(self) -> int:
        """
        Return None for no max intput count

        Returns
        -------
        int
            input_count

        """
        return None

    def add_done_callback(self, cb):
        """
        Appends callbacks when input is completed.

        Parameters
        ----------
        cb : function
            func

        """
        self._done_callbacks.append(cb)

    @typing.final
    async def build(self, input_stream: StreamPair) -> StreamPair:
        """
        This build method is a specialization of the `StreamWrapper.build` method. It allows derived sources
        to easily set up debug sink functions. Should not be overridden. Instead implement the abstract `_build` method.

        Parameters
        ----------
        input_stream : StreamPair
            A tuple containing the input `streamz.Stream` object and the message data type.

        Returns
        -------
        StreamPair
            A tuple containing the output `streamz.Stream` object from this stage and the message data type.

        """
        assert input_stream is None, "Sources shouldnt have input streams"

        self._input_stream = None

        output = await self._build()

        self._output_stream = output[0]

        if (self._post_sink_fn is not None):
            self._output_stream.gather().sink(self._post_sink_fn)

        return output

    @abstractmethod
    async def _build(self) -> StreamPair:
        """
        Abstract method all derived Source classes should implement. Returns the same value as `build`

        Returns
        -------

        StreamPair: 
            A tuple containing the output `streamz.Stream` object from this stage and the message data type.
        """

        pass


class Stage(StreamWrapper):
    """
    This class serves as the base for all pipeline stage implementations that are not source objects.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._pre_sink_fn: typing.Callable[[typing.Any], None] = None
        self._post_sink_fn: typing.Callable[[typing.Any], None] = None

        # Derived classes should set this to true to log timestamps in debug builds
        self._should_log_timestamps = False

    @abstractmethod
    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned. Derived classes should override this method. An
        error will be generated if the input types to the stage do not match one of the available types
        returned from this method.

        Returns
        -------
        typing.Tuple
            Accepted input types

        """
        pass

    async def build(self, input_stream: StreamPair) -> StreamPair:
        """
        This build method is a specialization of the `StreamWrapper.build` method. It allows derived Stage classes to quickly access input/output streams, set up debugging, and checks for the correct input types. Should not be overridden. Instead implement the abstract `_build` method.

        Parameters
        ----------
        input_stream : StreamPair
            A tuple containing the input `streamz.Stream` object and the message data type.

        Returns
        -------
        StreamPair
            A tuple containing the output `streamz.Stream` object from this stage and the message data type.

        """
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
        """
        Abstract method all derived Stage classes should implement. Has the same signature as the `build`
        method. All stage initializeation and contruction of stream objects should happen in this method.

        Parameters
        ----------
        input_stream : StreamPair
            A tuple containing the input `streamz.Stream` object and the message data type.

        Returns
        -------
        StreamPair: 
            A tuple containing the output `streamz.Stream` object from this stage and the message data type.
        """

        pass

    def on_start(self):
        """
        This function can be overridden to add usecase-specific implementation at the start of any stage in
        the pipeline.
        """
        pass

    def _on_complete(self, stream: Stream):

        logger.debug("Stage Complete: {}".format(self.name))


class Pipeline():
    """
    Class for building your pipeline. A pipeline for your use case can be constructed by first adding a
    `Source` via `set_source` then any number of downstream `Stage` classes via `add_stage`. The order stages
    are added with `add_stage` determines the order in which stage executions are carried out. You can use
    stages included within Morpheus or your own custom-built stages.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
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
        """
        Returns thread pool instance.

        Returns
        -------
        concurrent.futures.ThreadPoolExecutor
            thread pool instance to run on multi-thread execution.

        """
        return self._thread_pool

    def _add_id_col(self, x: cudf.DataFrame):

        # Data in stream is cudf Dataframes at this point. We need an ID column before continuing
        x.insert(0, 'ID', range(self._id_counter, self._id_counter + len(x)))
        self._id_counter += len(x)

        return x

    def set_source(self, source: SourceStage):
        """
        Set a pipeline's source stage to consume messages before it begins executing stages. This must be
        called once before `build_and_start`.

        Parameters
        ----------
        source : morpheus.pipeline.SourceStage
            The source stage wraps the implementation in a stream that allows it to read from Kafka or a file.

        """
        self._source_stage = source
        source._pipeline = self

    def add_stage(self, stage: Stage):
        """
        Add stages to the pipeline. All `Stage` classes added with this method will be executed sequentially
        inthe order they were added

        Parameters
        ----------
        stage : morpheus.pipeline.Stage
            The stage object to add. It cannot be already added to another `Pipeline` object.

        """
        self._stages.append(stage)
        stage._pipeline = self

    async def build_and_start(self):
        """
        This function sequentially activates all of the Morpheus pipeline stages passed by the users to
        execute a pipeline. For the `Source` and all added `Stage` objects, `StreamWrapper.build` will be
        called sequentially to construct the pipeline.

        Once the pipeline has been constructed, this will start the pipeline by calling `Source.start` on the
        source object.
        """
        if (self._use_dask):
            logger.info("====Launching Dask====")
            from distributed import Client
            self._client: Client = await Client(loop=IOLoop.current(), processes=True, asynchronous=True)

        logger.info("====Building Pipeline====")

        source_stream_pair = await self._source_stage.build(None)
        self._source_stream = source_stream_pair[0]

        self._source_stage.add_done_callback(self._on_input_complete)

        source_stream_pair[0].sink(self._on_start)

        # Add the ID single threaded
        current_stream_and_type = source_stream_pair
        # current_stream_and_type = current_stream_and_type[0].map(self._add_id_col), current_stream_and_type[1]

        logger.info("Added source: {} -> {}".format(self._source_stage.name, str(current_stream_and_type[1])))

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

            logger.info("Added stage: {} -> {}".format(s.name, str(current_stream_and_type[1])))

        logger.info("====Starting Inference====")

        self._source_stream.start()

    def _on_start(self, _):

        logger.debug("Starting! Time: {}".format(time.time()))

        # Loop over all stages and call on_start if it exists
        for s in self._stages:
            s.on_start()

    def _on_input_complete(self):
        logger.debug("All Input Complete")

    def run(self):
        """
        This function makes use of asyncio features to keep the pipeline running indefinitely.
        """
        loop = asyncio.get_event_loop()

        def error_handler(l, context: dict):

            msg = "Unhandled exception in async loop! Exception: \n{}".format(context["message"])
            exception = context.get("exception", Exception())

            logger.critical(msg)

        loop.set_exception_handler(error_handler)

        asyncio.ensure_future(self.build_and_start())

        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
            logger.debug("Exited")
