import logging
import os
import time
import typing

import numpy as np
import pandas as pd
import srf
from srf.core import operators as ops

import cudf

from morpheus.config import Config
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

# Setup conda environment
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': ['python={}'.format('3.8'), 'pip'],
    'pip': ['mlflow', 'dfencoder'],
    'name': 'mlflow-env'
}

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPRollingWindowStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 window_duration: str,
                 min_history: int,
                 max_history: typing.Union[int, str],
                 s3_cache_dir: str = "./.s3_cache"):
        super().__init__(c)

        self._window_duration = window_duration
        self._min_history = min_history
        self._max_history = max_history
        self._s3_cache_dir = s3_cache_dir

        # Map of user ids to total number of messages. Keeps indexes monotonic and increasing per user
        self._user_cache_map: typing.Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "dfp-rolling-window"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (UserMessageMeta, )

    def _trim_dataframe(self, df: pd.DataFrame):

        if (self._max_history is None):
            return df

        # See if max history is an int
        if (isinstance(self._max_history, int)):
            return df.tail(self._max_history)

        # If its a string, then its a duration
        if (isinstance(self._max_history, str)):
            # Get the latest timestamp
            latest = df["timestamp"].max()

            time_delta = pd.Timedelta(self._max_history)

            # Calc the earliest
            earliest = latest - time_delta

            return df[df['timestamp'] >= earliest]

        raise RuntimeError("Unsupported max_history")

    def _build_window(self, message: UserMessageMeta) -> UserMessageMeta:

        user_id = message.user_id

        if (user_id in self._user_cache_map):
            # Load any existing data
            existing_df = pd.read_pickle(self._user_cache_map.get(user_id))
        else:
            # Determine cache location
            cache_location = os.path.join(self._s3_cache_dir, "user-data", f"{user_id}.pkl")

            # Ensure the folder exists
            os.makedirs(os.path.dirname(cache_location), exist_ok=True)

            self._user_cache_map[user_id] = cache_location

            existing_df = pd.DataFrame()

        # Concat the incoming data with the old data
        concat_df = pd.concat([existing_df, message.df])

        concat_df.sort_values("timestamp", inplace=True, ignore_index=True)

        # Trim based on the rolling criteria
        concat_df = self._trim_dataframe(concat_df)

        # Save a new copy of the history
        concat_df.to_pickle(self._user_cache_map.get(user_id))

        # Exit early if we dont have enough data
        if (len(concat_df) < self._min_history):
            return None

        # Otherwise return a new message
        return UserMessageMeta(concat_df, user_id=message.user_id)

    def on_data(self, message: UserMessageMeta):
        start_time = time.time()

        result = self._build_window(message)

        duration = (time.time() - start_time) * 1000.0

        if (result is not None):
            logger.debug(
                "Rolling window complete for %s in %s ms. Input: %s rows from %s to %s. Output: %s rows from %s to %s",
                message.user_id,
                duration,
                len(message.df),
                message.df["timestamp"].min(),
                message.df["timestamp"].max(),
                len(result.df),
                result.df["timestamp"].min(),
                result.df["timestamp"].max(),
            )
        # else:
        #     logger.debug(
        #         "Rolling window complete for %s in %s ms. Input: %s rows from %s to %s. Output: None",
        #         message.user_id,
        #         duration,
        #         len(message.df),
        #         message.df["timestamp"].min(),
        #         message.df["timestamp"].max(),
        #     )

        return result

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, UserMessageMeta
