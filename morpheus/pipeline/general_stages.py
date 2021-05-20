import time
import typing
from functools import reduce

import cudf
import cupy as cp
from morpheus.config import Config
from morpheus.pipeline import Stage
from morpheus.pipeline.messages import MultiMessage, MultiResponseMessage
from morpheus.pipeline.pipeline import StreamPair
from streamz.core import Stream
from tqdm import tqdm


class BufferStage(Stage):
    """
    The input messages are buffered by this stage class for faster access to downstream stages. Allows
    upstream stages to run faster than downstream stages.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config, count: int = 1000):
        super().__init__(c)

        self._buffer_count = count

    @property
    def name(self) -> str:
        return "buffer"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types

        """
        return (typing.Any, )

    async def _build(self, input_stream: StreamPair) -> StreamPair:
        return input_stream[0].buffer(self._buffer_count), input_stream[1]


class DelayStage(Stage):
    """
    Delay stage class. Used to buffer all inputs until the timeout duration is hit. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config, duration: str):
        super().__init__(c)

        self._duration = duration

    @property
    def name(self) -> str:
        return "delay"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types

        """
        return (typing.Any, )

    async def _build(self, input_stream: StreamPair) -> StreamPair:
        return input_stream[0].time_delay(self._duration), input_stream[1]


class TriggerStage(Stage):
    """
    This stage will buffer all inputs until the source stage is complete. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "trigger"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types

        """
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
    """
    Monitor stage used to monitor stage performance metrics using Tqdm. Each Monitor Stage will represent one
    line in the console window showing throughput statistics. Can be set up to show an instantaneous
    throughput or average input.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    description : str
        Name to show for this Monitor Stage in the console window
    smoothing : int
        Smoothing parameter to determine how much the throughput should be averaged. 0 = Instantaneous, 1 =
        Average.
    unit : str
        Units to show in the rate value.
    determine_count_fn : typing.Callable[[typing.Any], int]
        Custom function for determining the count in a message. Gets called for each message. Allows for
        correct counting of batched and sliced messages.

    """
    def __init__(self,
                 c: Config,
                 description: str = "Progress",
                 smoothing: int = 0.05,
                 unit="messages",
                 determine_count_fn: typing.Callable[[typing.Any], int] = None):
        super().__init__(c)

        self._progress: tqdm = None

        self._description = description
        self._smoothing = smoothing
        self._unit = unit

        # self._pre_sink_fn = self.pre_timestamps
        self._post_sink_fn = self._progress_sink

        self._determine_count_fn = determine_count_fn

    @property
    def name(self) -> str:
        return "monitor"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types

        """
        return (typing.Any, )

    def on_start(self):

        self._progress = tqdm(desc=self._description,
                              smoothing=self._smoothing,
                              dynamic_ncols=True,
                              unit=self._unit,
                              mininterval=0.25,
                              maxinterval=1.0)

        self._progress.reset()

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

        # # This starts the timer on the first message. Otherwise it sits from creation with no updates
        # if (self._progress.n == 0):
        #     self._progress.unpause()

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
    """
    Add classification labels based on probabilities calculated in inference stage. Uses default threshold of
    0.5 for predictions.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    threshold : float
        Threshold to classify, default is 0.5

    """
    def __init__(self, c: Config, threshold: float = 0.5):
        super().__init__(c)

        self._threshold = threshold

    @property
    def name(self) -> str:
        return "add-class"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[MultiResponseMessage, ]
            Accepted input types

        """
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
    """
    This Stage class is used to filter results based on a given criteria.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    threshold : float
        Threshold to classify, default is 0.5

    """
    def __init__(self, c: Config, threshold: float = 0.5):
        super().__init__(c)

        # Probability to consider a detection
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "filter"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[MultiResponseMessage, ]
            Accepted input types

        """
        return (MultiResponseMessage, )

    def filter(self, x: MultiResponseMessage) -> typing.List[MultiResponseMessage]:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : morpheus.messages.MultiResponseMessage
            MultiResponseMessage

        Returns
        -------
        typing.List[MultiResponseMessage]
            list of filtered messages

        """
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
