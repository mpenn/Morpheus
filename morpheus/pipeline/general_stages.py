from functools import reduce
from morpheus.pipeline.pipeline import StreamPair
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
import morpheus.utils.async_map
class BufferStage(Stage):
    def __init__(self, c: Config, count: int = 1000):
        super().__init__(c)

        self._buffer_count = count

    @property
    def name(self) -> str:
        return "buffer"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    async def _build(self, input_stream: StreamPair) -> StreamPair:
        return input_stream[0].buffer(self._buffer_count), input_stream[1]

class DelayStage(Stage):
    def __init__(self, c: Config, duration: str):
        super().__init__(c)

        self._duration = duration

    @property
    def name(self) -> str:
        return "delay"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    async def _build(self, input_stream: StreamPair) -> StreamPair:
        return input_stream[0].time_delay(self._duration), input_stream[1]

class TriggerStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "trigger"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        collector = stream.collect()

        def flush_input(_: Stream):
            collector.flush()

        stream.add_done_callback(flush_input)

        stream = collector.flatten()

        return stream, input_stream[1]


class MonitorStage(Stage):
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

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        # input_stream[0].add_done_callback(self._refresh_progress)

        return input_stream

    def _refresh_progress(self, _):
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

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Convert the messages to rows of strings
        stream = stream.async_map(self._add_labels, executor=self._pipeline.thread_pool)

        # Return input unchanged
        return stream, MultiResponseMessage

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

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Reduce messages to only have detections
        stream = stream.async_map(self.filter, executor=self._pipeline.thread_pool)

        # Convert list back to single MultiResponseMessage
        stream = stream.flatten()

        # Filter out empty message groups
        stream = stream.filter(lambda x: x.count > 0)

        return stream, MultiResponseMessage

