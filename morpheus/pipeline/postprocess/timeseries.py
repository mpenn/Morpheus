# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import datetime as dt
import logging
import typing
from collections import deque
from math import ceil

import cupy as cp
import neo
import pandas as pd
from neo.core import operators as ops
from streamz.core import Stream
from streamz.core import sync

from morpheus.config import Config
from morpheus.pipeline.messages import MultiResponseAEMessage
from morpheus.pipeline.messages import MultiResponseMessage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)


def round_seconds(obj: pd.Timestamp) -> pd.Timestamp:
    return obj.round(freq="S")


def calc_bin(obj: pd.Timestamp, t0: pd.Timestamp, resolution_sec: float) -> int:
    return round((round_seconds(obj) - t0).total_seconds()) // resolution_sec


def zscore(data):
    """
    Calculate z score of cupy.ndarray
    """
    mu = cp.mean(data)
    std = cp.std(data)
    return cp.abs(data - mu) / std


def to_periodogram(signal_cp: cp.ndarray):
    """
    Returns periodogram of signal for finding frequencies that have high energy.
    :param signal: signal (time domain)
    :type signal: cudf.Series
    :return: CuPy array representing periodogram
    :rtype: cupy.ndarray
    """

    std_dev = cp.std(signal_cp)

    # standardize the signal
    if (std_dev != 0.0):
        signal_cp_std = (signal_cp - cp.mean(signal_cp)) / std_dev
    else:
        # Otherwise they are all 0 sigma away
        signal_cp_std = signal_cp - cp.mean(signal_cp)

    # take fourier transform of signal
    FFT_data = cp.fft.fft(signal_cp_std)

    # create periodogram
    prdg = (1 / len(signal_cp)) * ((cp.absolute(FFT_data))**2)

    return prdg


def fftAD(signalvalues: cp.ndarray, p=90, zt=8, lowpass=None):
    """
    Detect anomalies with fast fourier transform

    Parameters
    ----------
    vals : cupy.ndarray or numpy.ndarray
        Values of time signal (real valued)
    p : int (default = 90)
        Filtering percentile for spectral density based filtering
    zt : float (default = 8)
        Z-score threshold, can be tuned for datasets and sensitivity
    lowpass : int (default = None)
        Filtering percentile for frequency based filtering
    visualize : boolean (default = False)
        Whether to visualize signal and filtering

    Returns
    -------
    is_anomaly: cupy.ndarray
        binary vector whether each point is anomalous
    """
    periodogram = to_periodogram(signalvalues)
    periodogram = periodogram[:len(signalvalues) // 2 + 1]

    indices_mask = cp.zeros_like(periodogram, dtype=bool)

    # lowpass: percentile to keep
    if lowpass:
        freqs = cp.arange(len(periodogram))
        bar = int(cp.percentile(freqs, lowpass))
        indices_mask[bar:] = True
    # p: percentile to delete
    else:
        threshold = cp.percentile(periodogram, p).item()

        indices_mask = (periodogram < threshold)

    rft = cp.fft.rfft(signalvalues, n=len(signalvalues))
    rft[indices_mask] = 0
    recon = cp.fft.irfft(rft, n=len(signalvalues))

    err = (abs(recon - signalvalues))

    z = zscore(err)

    return cp.arange(len(signalvalues))[z >= zt]


@Stream.register_api()
class delayed_send(Stream):

    _graphviz_shape = 'diamond'

    def __init__(self,
                 upstream,
                 process_func: typing.Callable[[typing.Any, bool], typing.Optional[typing.Any]],
                 **kwargs):

        self.metadata_buffer = deque()

        self._process_func = process_func

        Stream.__init__(self, upstream, **kwargs)

    async def update(self, x, who=None, metadata=None):

        ret = []

        # Save the metadata in case it gets held
        self._retain_refs(metadata)

        if not isinstance(metadata, list):
            metadata = [metadata]

        self.metadata_buffer.append(metadata)

        # Call the process function to see if we have any values to emit
        while to_process := self._process_func(x, False):

            # Value to emit, pop the buffer and emit
            to_process_meta = self.metadata_buffer.popleft()

            r = await self._emit(to_process, metadata=to_process_meta)

            self._release_refs(to_process_meta)

            ret.append(r)

            # Clear x so its not re-added
            x = None

        return ret

    def on_complete(self):
        async def flush():
            while to_process := self._process_func(None, True):

                to_process_meta = self.metadata_buffer.popleft()

                await self._emit(to_process, metadata=to_process_meta)

                self._release_refs(to_process_meta)

        # Run the flush function syncronously
        sync(self.loop, flush)

        return super().on_complete()


@Stream.register_api()
class variable_send(Stream):

    _graphviz_shape = 'diamond'

    def __init__(self, upstream, process_func: typing.Callable[[typing.Any, bool], typing.List[typing.Any]], **kwargs):

        self.metadata_buffer = deque()

        self._process_func = process_func

        Stream.__init__(self, upstream, **kwargs)

    async def update(self, x, who=None, metadata=None):

        # Save the metadata in case it gets held
        self._retain_refs(metadata)

        self.metadata_buffer.append(metadata)

        to_process = self._process_func(x, False)

        ret = []

        # Iterate on any objects we got back
        for tp in to_process:
            # Value to emit, pop the buffer and emit
            tp_meta = self.metadata_buffer.popleft()

            r = await self._emit(tp, metadata=tp_meta)

            self._release_refs(tp_meta)

            ret.append(r)

        return ret

    async def _on_completed(self):

        # Need to call the process function to flush out remaining items before completing
        async def flush():

            to_process = self._process_func(None, True)

            # Iterate on any objects we got back
            for tp in to_process:
                # Value to emit, pop the buffer and emit
                tp_meta = self.metadata_buffer.popleft()

                await self._emit(tp, metadata=tp_meta)

                self._release_refs(tp_meta)

        # Run the flush function syncronously
        await flush()

        # Call base
        return super()._on_completed()


@dataclasses.dataclass
class TimeSeriesAction:
    perform_calc: bool = False
    window: pd.DataFrame = None
    window_start: dt.datetime = None
    window_end: dt.datetime = None

    send_message: bool = False
    message: MultiResponseMessage = None


class UserTimeSeries(object):
    def __init__(self,
                 user_id: str,
                 resolution: str,
                 min_window: str,
                 hot_start: bool,
                 cold_end: bool,
                 filter_percent: float,
                 zscore_threshold: float) -> None:
        super().__init__()

        self._user_id = user_id

        # Size of bins
        self._resolution_sec = int(round(pd.Timedelta(resolution).total_seconds()))

        # Set the min window to a week to start
        self._min_window_sec = pd.Timedelta(min_window).total_seconds()
        self._half_window_bins = int(ceil(self._min_window_sec / self._resolution_sec))

        # Flag indicating if the warmup period has completed
        self._is_warm = hot_start

        # Flag indicating whether to allow window violation during shutdown
        self._cold_end = cold_end

        self._filter_percent = filter_percent
        self._zscore_threshold = zscore_threshold

        # Keep track of the max index we have seen. All dataframes must share a single index to confirm order
        self._processed_index = 0
        self._holding_timestamps = deque()

        # Stateful members
        self._pending_messages: deque[MultiResponseMessage] = deque()  # Holds the existing messages pending
        self._timeseries_data: pd.DataFrame = pd.DataFrame(columns=["event_dt"])  # Holds all available timeseries data

        self._t0_epoch: float = None

    def _calc_bin_series(self, t: pd.Series) -> pd.Series:

        return round((t.dt.round(freq="S") - self._t0_epoch).dt.total_seconds()).astype(int) // self._resolution_sec

    def _calc_outliers(self, action: TimeSeriesAction):

        # Subtract one bin to add buffers on either side. This makes the calculation of the bins easier
        window_start = action.window_start - 1
        window_end = action.window_end

        signal_cp, signal_bins = cp.histogram(cp.array(action.window["event_bin"]),
                                              bins=cp.arange(window_start, window_end + 1, 1))

        # TODO(MDD): Take this out after testing
        assert cp.sum(signal_cp) == len(action.window), "All points in window are not accounted for in histogram"

        is_anomaly = fftAD(signal_cp, p=self._filter_percent, zt=self._zscore_threshold)

        if (len(is_anomaly) > 0):

            # Convert back to bins
            is_anomaly = cp.choose(is_anomaly, signal_bins)

            anomalies = action.window["event_bin"].isin(is_anomaly.get())

            # Set the anomalies by matching indexes
            action.message.set_meta("ts_anomaly", anomalies)

            # Find anomalies that are in the active message
            paired_anomalies = anomalies[anomalies == True].index.intersection(action.message.get_meta().index)

            # Return the anomalies for priting. But only if the current message has anomalies that will get flagged
            if (len(paired_anomalies) > 0):
                return self._t0_epoch + pd.to_timedelta(
                    (action.window.loc[paired_anomalies]["event_bin"].unique() * self._resolution_sec), unit='s')

        return None

    def _determine_action(self, is_complete: bool) -> typing.Optional[TimeSeriesAction]:

        # Stop processing on empty queue
        if (len(self._pending_messages) == 0):
            return None

        # Note: We calculate everything in bins to ensure 1) Full bins, and 2) Even binning
        timeseries_start = self._timeseries_data["event_bin"].iloc[0]
        timeseries_end = self._timeseries_data["event_bin"].iloc[-1]

        # Peek the front message
        x: MultiResponseMessage = self._pending_messages[0]

        # Get the first message timestamp
        message_start = calc_bin(x.get_meta("event_dt").iloc[0], self._t0_epoch, self._resolution_sec)
        message_end = calc_bin(x.get_meta("event_dt").iloc[-1], self._t0_epoch, self._resolution_sec)

        window_start = message_start - self._half_window_bins
        window_end = message_end + self._half_window_bins

        # Check left buffer
        if (timeseries_start > window_start):
            # logger.debug("Warming up.    TS: %s, WS: %s, MS: %s, ME: %s, WE: %s, TE: %s. Delta: %s",
            #              timeseries_start._repr_base,
            #              window_start._repr_base,
            #              message_start._repr_base,
            #              message_end._repr_base,
            #              window_end._repr_base,
            #              timeseries_end._repr_base,
            #              timeseries_start - window_start)

            # Not shutting down and we arent warm, send through
            if (not self._is_warm and not is_complete):
                return TimeSeriesAction(send_message=True, message=self._pending_messages.popleft())

        self._is_warm = True

        # Check the right buffer
        if (timeseries_end < window_end):
            # logger.debug("Filling front. TS: %s, WS: %s, MS: %s, ME: %s, WE: %s, TE: %s. Delta: %s",
            #              timeseries_start._repr_base,
            #              window_start._repr_base,
            #              message_start._repr_base,
            #              message_end._repr_base,
            #              window_end._repr_base,
            #              timeseries_end._repr_base,
            #              window_end - timeseries_end)

            if (not is_complete):
                # Not shutting down, so hold message
                return None
            elif (is_complete and self._cold_end):
                # Shutting down and we have a cold ending, just empty the message
                return TimeSeriesAction(send_message=True, message=self._pending_messages.popleft())
            else:
                # Shutting down and hot end
                # logger.debug("Hot End. Processing. TS: %s", timeseries_start._repr_base)
                pass

        # By this point we have both a front and back buffer. So get ready for a calculation
        # logger.debug("Perform Calc.  TS: %s, WS: %s, MS: %s, ME: %s, WE: %s, TE: %s.",
        #              timeseries_start._repr_base,
        #              window_start._repr_base,
        #              message_start._repr_base,
        #              message_end._repr_base,
        #              window_end._repr_base,
        #              timeseries_end._repr_base)

        # First, remove elements in the front queue that are too old
        self._timeseries_data.drop(self._timeseries_data[self._timeseries_data["event_bin"] < window_start].index,
                                   inplace=True)

        # Now grab timestamps within the window
        window_timestamps_df = self._timeseries_data[self._timeseries_data["event_bin"] <= window_end]

        # Return info to perform calc
        return TimeSeriesAction(perform_calc=True,
                                window=window_timestamps_df,
                                window_start=window_start,
                                window_end=window_end,
                                send_message=True,
                                message=self._pending_messages.popleft())

    def _calc_timeseries(self, x: MultiResponseMessage, is_complete: bool):

        if (x is not None):

            # Ensure that we have the meta column set for all messages
            x.set_meta("ts_anomaly", False)

            # Save this message in the pending queue
            self._pending_messages.append(x)

            new_timedata = x.get_meta(["event_dt"])

            # Save this message event times in the event list. Ensure the values are always sorted
            self._timeseries_data = pd.concat([self._timeseries_data, new_timedata]).sort_index()

            # Check to see if these are out of order. If they are, just return empty and try again next item
            if (len(
                    pd.RangeIndex(start=self._processed_index, stop=self._timeseries_data.index[-1] + 1,
                                  step=1).difference(self._timeseries_data.index)) > 0):
                return []

            # Save the new max index
            self._processed_index = self._timeseries_data.index[-1] + 1

        # If this is our first time data, set the t0 time
        if (self._t0_epoch is None):
            self._t0_epoch = self._timeseries_data["event_dt"].iloc[0]

            # TODO(MDD): Floor to the day to unsure all buckets are always aligned with val data
            self._t0_epoch = self._t0_epoch.floor(freq="D")

        # Calc the bins for the timeseries data
        self._timeseries_data["event_bin"] = self._calc_bin_series(self._timeseries_data["event_dt"])

        # At this point there are 3 things that can happen
        # 1. We are warming up to build a front buffer. Save the current message times and send the message on
        # 2. We are warmed up and building a back buffer, Save the current message, and message times. Hold the message
        # 3. We have a front and back buffer. Perform the calc, identify outliers, and send the message on. Repeat

        output_messages = []

        # Now we calc if we have enough range
        while action := self._determine_action(is_complete):

            if (action.perform_calc):
                # Actually do the calc
                anomalies = self._calc_outliers(action)

                if (anomalies is not None):
                    if (is_complete):
                        logger.debug("Found anomalies (Shutdown): %s", list(anomalies))
                    else:
                        logger.debug("Found anomalies: %s", list(anomalies))

            if (action.send_message):
                # Now send the message
                output_messages.append(action.message)

        return output_messages


class TimeSeriesStage(SinglePortStage):
    """
    Perform time series anomaly detection and add prediction. Uses default resolution of
    "hour".

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    resolution : str
        Time series resolution. Logs will be binned into groups of this size. Uses the pandas time delta format, i.e.
        '10m' for 10 minutes
    min_window : str
        Minimum window on either side of a log necessary for calculation. Logs will be skipped during a warmup phase
        while this window is filled. Uses the pandas time delta format, i.e. '10m' for 10 minutes
    hot_start : bool, default = False
        This flag prevents the stage from ignoring messages during a warm up phase while the min_window is filled.
        Enabling 'hot_start' will run calculations on all messages even if the min_window is not satisfied on both
        sides, i.e. during startup or teardown. This is likely to increase the number of false positives but can be
        helpful for debugging and testing on small datasets.
    cold_end : bool, default = True
        This flag is the inverse of `hot_start` and prevents ignoring messages that don't satisfy the `min_window`
        during the shutdown phase. This is likely to increase the number of false positives but can be helpful for
        debugging and testing on small datasets.
    filter_percent : float, default = 90
        The percent of timeseries samples to remove from the inverse FFT for spectral density filtering.
    zscore_threshold : float, default = 8.0
        The z-score threshold required to flag datapoints. The value indicates the number of standard deviations from
        the mean that is required to be flagged. Increasing this value will decrease the number of detections.
    """
    def __init__(self,
                 c: Config,
                 resolution: str,
                 min_window: str,
                 hot_start: bool,
                 cold_end: bool,
                 filter_percent: float,
                 zscore_threshold: float):
        super().__init__(c)

        self._feature_length = c.feature_length

        self._resolution = resolution
        self._min_window = min_window
        self._hot_start = hot_start
        self._cold_end = cold_end
        self._filter_percent = filter_percent
        self._zscore_threshold = zscore_threshold

        self._timeseries_per_user: typing.Dict[str, UserTimeSeries] = {}

    @property
    def name(self) -> str:
        return "timeseries"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[MultiResponseMessage, ]
            Accepted input types

        """
        return (MultiResponseMessage, )

    def _call_timeseries_user(self, x: MultiResponseAEMessage):

        if (x.user_id not in self._timeseries_per_user):
            self._timeseries_per_user[x.user_id] = UserTimeSeries(user_id=x.user_id,
                                                                  resolution=self._resolution,
                                                                  min_window=self._min_window,
                                                                  hot_start=self._hot_start,
                                                                  cold_end=self._cold_end,
                                                                  filter_percent=self._filter_percent,
                                                                  zscore_threshold=self._zscore_threshold)

        return self._timeseries_per_user[x.user_id]._calc_timeseries(x, False)

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = input_stream[1]

        def node_fn(input: neo.Observable, output: neo.Subscriber):
            def on_next(x: MultiResponseAEMessage):

                message_list: typing.List[MultiResponseMessage] = self._call_timeseries_user(x)

                return message_list

            def on_completed():

                to_send = []

                for ts in self._timeseries_per_user.values():
                    message_list: typing.List[MultiResponseMessage] = ts._calc_timeseries(None, True)

                    to_send = to_send + message_list

                return to_send if len(to_send) > 0 else None

            input.pipe(ops.map(on_next),
                       ops.filter(lambda x: len(x) > 0),
                       ops.on_completed(on_completed),
                       ops.flatten()).subscribe(output)

        stream = seg.make_node_full(self.unique_name, node_fn)

        seg.make_edge(input_stream[0], stream)

        return stream, out_type
