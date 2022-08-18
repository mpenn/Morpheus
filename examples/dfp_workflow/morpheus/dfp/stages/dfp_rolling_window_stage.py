# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
import os
import pickle
import time
import typing
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import srf
from srf.core import operators as ops

import cudf

from morpheus.config import Config
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..utils.logging_timer import log_time

# Setup conda environment
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': ['python={}'.format('3.8'), 'pip'],
    'pip': ['mlflow', 'dfencoder'],
    'name': 'mlflow-env'
}

logger = logging.getLogger("morpheus.{}".format(__name__))


@dataclasses.dataclass
class CachedUserWindow:
    user_id: str
    cache_location: str
    timestamp_column: str = "timestamp"
    total_count: int = 0
    count: int = 0
    min_epoch: datetime = datetime(1970, 1, 1)
    max_epoch: datetime = datetime(1970, 1, 1)
    last_train_count: int = 0
    last_train_epoch: datetime = None

    _trained_rows: pd.Series = dataclasses.field(init=False, repr=False, default_factory=pd.DataFrame)
    _df: pd.DataFrame = dataclasses.field(init=False, repr=False, default_factory=pd.DataFrame)

    def append_dataframe(self, incoming_df: pd.DataFrame):

        # # Get the row hashes
        # row_hashes = pd.util.hash_pandas_object(incoming_df)

        # Filter the incoming df by epochs later than the current max_epoch
        filtered_df = incoming_df[incoming_df["timestamp"] > self.max_epoch]

        # Set the filtered index
        filtered_df.index = range(self.total_count, self.total_count + len(filtered_df))

        # Append just the new rows
        self._df = pd.concat([self._df, filtered_df])

        self.total_count += len(filtered_df)
        self.count = len(self._df)

        if (len(self._df) > 0):
            self.min_epoch = self._df[self.timestamp_column].min()
            self.max_epoch = self._df[self.timestamp_column].max()

    def get_train_df(self, max_history) -> pd.DataFrame:

        new_df = self.trim_dataframe(self._df, max_history=max_history)

        self.last_train_count = self.total_count
        self.last_train_epoch = datetime.now()

        self._df = new_df

        if (len(self._df) > 0):
            self.min_epoch = self._df[self.timestamp_column].min()
            self.max_epoch = self._df[self.timestamp_column].max()

        return new_df

    def save(self):

        # Make sure the directories exist
        os.makedirs(os.path.dirname(self.cache_location), exist_ok=True)

        with open(self.cache_location, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def trim_dataframe(df: pd.DataFrame,
                       max_history: typing.Union[int, str],
                       timestamp_column: str = "timestamp") -> pd.DataFrame:
        if (max_history is None):
            return df

        # See if max history is an int
        if (isinstance(max_history, int)):
            return df.tail(max_history)

        # If its a string, then its a duration
        if (isinstance(max_history, str)):
            # Get the latest timestamp
            latest = df[timestamp_column].max()

            time_delta = pd.Timedelta(max_history)

            # Calc the earliest
            earliest = latest - time_delta

            return df[df['timestamp'] >= earliest]

        raise RuntimeError("Unsupported max_history")

    @staticmethod
    def load(cache_location: str) -> "CachedUserWindow":

        with open(cache_location, "rb") as f:
            return pickle.load(f)


class DFPRollingWindowStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 min_history: int,
                 min_increment: int,
                 max_history: typing.Union[int, str],
                 cache_dir: str = "./.cache/dfp"):
        super().__init__(c)

        self._min_history = min_history
        self._min_increment = min_increment
        self._max_history = max_history
        self._cache_dir = os.path.join(cache_dir, "rolling-user-data")

        # Map of user ids to total number of messages. Keeps indexes monotonic and increasing per user
        self._user_cache_map: typing.Dict[str, CachedUserWindow] = {}

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
            latest = df[self._config.ae.timestamp_column_name].max()

            time_delta = pd.Timedelta(self._max_history)

            # Calc the earliest
            earliest = latest - time_delta

            return df[df['timestamp'] >= earliest]

        raise RuntimeError("Unsupported max_history")

    @contextmanager
    def _get_user_cache(self, user_id: str):

        # Determine cache location
        cache_location = os.path.join(self._cache_dir, f"{user_id}.pkl")

        user_cache = None

        try:
            if (os.path.exists(cache_location)):
                # Try to load any existing window
                user_cache = CachedUserWindow.load(cache_location=cache_location)
        except:
            logger.warning("Error loading window cache at %s", cache_location, exc_info=True)

        if (user_cache is None):
            user_cache = CachedUserWindow(user_id=user_id, cache_location=cache_location)

        # if (user_id not in self._user_cache_map):
        #     # Determine cache location
        #     cache_location = os.path.join(self._cache_dir, f"{user_id}.pkl")

        #     # Ensure the folder exists
        #     os.makedirs(os.path.dirname(cache_location), exist_ok=True)

        #     self._user_cache_map[user_id] = CachedUserWindow(user_id=user_id, cache_location=cache_location)

        # user_cache = self._user_cache_map.get(user_id)

        yield user_cache

        # When it returns, make sure to save
        user_cache.save()

    def _build_window(self, message: UserMessageMeta) -> UserMessageMeta:

        user_id = message.user_id

        with self._get_user_cache(user_id) as user_cache:

            incoming_df = message.df
            # existing_df = user_cache.df

            user_cache.append_dataframe(incoming_df=incoming_df)

            # # For the incoming one, calculate the row hash to identify duplicate rows
            # incoming_df["_row_hash"] = pd.util.hash_pandas_object(incoming_df)

            # # Concat the incoming data with the old data
            # concat_df = pd.concat([existing_df, incoming_df])

            # # Drop any duplicates (only really happens when debugging)
            # concat_df = concat_df.drop_duplicates(subset=["_row_hash"], keep='first')

            # # Save the number of new rows here
            # new_row_count = len(concat_df) - len(existing_df)

            # # Finally, ensure we are sorted. This also resets the index
            # concat_df.sort_values(self._config.ae.timestamp_column_name, inplace=True, ignore_index=True)

            # # Trim based on the rolling criteria
            # concat_df = self._trim_dataframe(concat_df)

            # # Update cache object
            # user_cache.set_dataframe(concat_df,
            #                          new_row_count=new_row_count,
            #                          timestamp_column=self._config.ae.timestamp_column_name)

            # current_df_count = len(concat_df)

            # Exit early if we dont have enough data
            if (user_cache.count < self._min_history):
                return None

            # We have enough data, but has enough time since the last training taken place?
            if (user_cache.total_count - user_cache.last_train_count < self._min_increment):
                return None

            # Save the last train statistics
            train_df = user_cache.get_train_df(max_history=self._max_history)

            # Otherwise return a new message
            return UserMessageMeta(train_df, user_id=message.user_id)

    def on_data(self, message: UserMessageMeta):

        with log_time(logger.debug) as log_info:

            result = self._build_window(message)

            if (result is not None):

                log_info.set_log(
                    ("Rolling window complete for %s in {duration:0.2f} ms. "
                     "Input: %s rows from %s to %s. Output: %s rows from %s to %s"),
                    message.user_id,
                    len(message.df),
                    message.df[self._config.ae.timestamp_column_name].min(),
                    message.df[self._config.ae.timestamp_column_name].max(),
                    len(result.df),
                    result.df[self._config.ae.timestamp_column_name].min(),
                    result.df[self._config.ae.timestamp_column_name].max(),
                )
            else:
                # Dont print anything
                log_info.disable()

            return result

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, UserMessageMeta
