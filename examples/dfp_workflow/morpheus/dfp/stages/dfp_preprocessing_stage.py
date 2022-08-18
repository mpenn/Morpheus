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
import time
import typing

import pandas as pd
import srf
from srf.core import operators as ops

import cudf

from morpheus._lib.messages import MultiMessage
from morpheus.config import Config
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..utils.column_info import ColumnInfo
from ..utils.column_info import DataFrameInputSchema
from ..utils.column_info import process_dataframe

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPInferencePreprocessingStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 column_info: typing.List[ColumnInfo],
                 json_columns: typing.List[str],
                 return_format: str = "data"):
        super().__init__(c)

        # TODO(MDD): Make these all constructor params
        # Normalize step
        self._return_format = return_format

        self._json_columns = ["access_device", "application", "auth_device", "user"]
        self._json_columns = json_columns

        # Rename step
        self._rename_map = {
            'access_device.browser': 'accessdevicebrowser',
            'access_device.os': 'accessdeviceos',
            'user.name': 'username',
            "auth_device.location.city": "locationcity",
            "auth_device.name": "device",
        }
        self._remove_characters = "[_,.,{,},:]"

        # Get features step
        self._additional_columns = [self._config.ae.timestamp_column_name]

        self._column_info: typing.List[ColumnInfo] = column_info

    @property
    def name(self) -> str:
        return "dfp-preproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (cudf.DataFrame, )

    @staticmethod
    def _create_lonicrement(df):
        slot_list = []
        timeslots = df['time'].unique()
        for slot in timeslots:
            new_df = df[(df['time'] == slot)]
            new_df["locincrement"] = ((new_df['locationcity'].factorize()[0] + 1))
            slot_list.append(new_df)

        return pd.concat(slot_list)

    @staticmethod
    def _create_logcount(df):
        df["logcount"] = df.groupby('time').cumcount()

        return df

    def _normalize_dataframe(self, df):

        json_normalized = []
        remaining_columns = list(df.columns)

        for j_column in self._json_columns:

            if (j_column not in remaining_columns):
                continue

            normalized = pd.json_normalize(df[j_column])

            # Prefix the columns
            normalized.rename(columns={n: f"{j_column}.{n}" for n in normalized.columns}, inplace=True)

            # Reset the index otherwise there is a conflict
            normalized.reset_index(drop=True, inplace=True)

            json_normalized.append(normalized)

            # Remove from the list of remaining columns
            remaining_columns.remove(j_column)

        # Also need to reset the original index
        df.reset_index(drop=True, inplace=True)

        df_normalized = pd.concat([df[remaining_columns]] + json_normalized, axis=1)

        return df_normalized

    def _rename_columns(self, df):

        df.rename(columns=self._rename_map, inplace=True)

        df.columns = df.columns.str.replace(self._remove_characters, '')
        df.columns = df.columns.str.strip()

        return df

    def _process_columns_old(self, df):

        per_day = df[self._config.ae.timestamp_column_name].dt.to_period("D")

        # Create the per-user, per-day log count
        df["logcount"] = df.groupby([self._config.ae.userid_column_name, per_day]).cumcount()

        # # Create the per-user, per-day location increment count
        # df["locincrement"] = df.groupby([self._config.ae.userid_column_name,
        #                                  per_day])["locationcity"].transform(lambda x: pd.factorize(x)[0] + 1)

        return df

    def _process_single_column(self, ci: ColumnInfo, df: cudf.DataFrame):

        if (ci.input_name not in df.columns):
            # Generate warning?
            # Create empty column
            return cudf.Series(None, dtype=ci.dtype)

        input_col = df[ci.input_name]

        # TODO(MDD): Should we do some processing on converting types if they dont match?

        if (ci.process_column is not None):
            return ci.process_column(df)

        return input_col

    def _process_columns(self, df: cudf.DataFrame):

        output_df = cudf.DataFrame()

        # Iterate over the column info
        for ci in self._column_info:
            output_df[ci.name] = self._process_single_column(ci, df)

        return output_df

    def _get_features(self, df):

        final_columns = self._config.ae.feature_columns + self._additional_columns
        for feature in final_columns:
            if (feature not in df.columns):
                df[feature] = ""

        return df[final_columns]

    def process_features(self, df_in: cudf.DataFrame):
        if (df_in is None):
            return None

        start_time = time.time()

        # Step 1 is to normalize any columns
        df_processed = self._normalize_dataframe(df_in)

        # Step 2 is to process columns
        # df_processed = self._rename_columns(df_processed)
        df_processed = self._process_columns(df_processed)
        # df_processed = self._get_features(df_processed)

        duration = (time.time() - start_time) * 1000.0

        logger.debug("Preprocessed %s data for logs in %s to %s in %s ms",
                     len(df_in),
                     df_processed[self._config.ae.timestamp_column_name].min(),
                     df_processed[self._config.ae.timestamp_column_name].max(),
                     duration)

        return df_processed

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.process_features)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, cudf.DataFrame


class DFPTrainingPreprocessingStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 column_info: typing.List[ColumnInfo],
                 json_columns: typing.List[str],
                 frame_cache_dir: str = "./s3_cache",
                 generic_only: bool = True,
                 return_format: str = "data"):
        super().__init__(c)

        self._cache_ids = []
        self._generic_only = generic_only
        self._df_user_frames = pd.DataFrame(columns=("username", "frame_path"))
        self._cache_path = "preprocessing"
        self._s3_cache_dir = frame_cache_dir
        self._return_format = return_format

        self._json_columns = ["access_device", "application", "auth_device", "user"]
        self._json_columns = json_columns

        # Rename step
        self._rename_map = {
            'access_device.browser': 'accessdevicebrowser',
            'access_device.os': 'accessdeviceos',
            'user.name': 'username',
            "auth_device.location.city": "locationcity",
            "auth_device.name": "device",
        }
        self._remove_characters = "[_,.,{,},:]"

        # Get features step
        self._additional_columns = [self._config.ae.timestamp_column_name]

        self._column_info: typing.List[ColumnInfo] = column_info

    @property
    def name(self) -> str:
        return "dfp-preproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (cudf.DataFrame, )

    @staticmethod
    def _create_lonicrement(df):
        slot_list = []
        timeslots = df['time'].unique()
        for slot in timeslots:
            new_df = df[(df['time'] == slot)]
            new_df["locincrement"] = ((new_df['locationcity'].factorize()[0] + 1))
            slot_list.append(new_df)

        return pd.concat(slot_list)

    @staticmethod
    def _create_logcount(df):
        df["logcount"] = df.groupby('time').cumcount()

        return df

    def _normalize_dataframe(self, df):

        json_normalized = []
        remaining_columns = list(df.columns)

        for j_column in self._json_columns:

            if (j_column not in remaining_columns):
                continue

            normalized = pd.json_normalize(df[j_column])

            # Prefix the columns
            normalized.rename(columns={n: f"{j_column}.{n}" for n in normalized.columns}, inplace=True)

            # Reset the index otherwise there is a conflict
            normalized.reset_index(drop=True, inplace=True)

            json_normalized.append(normalized)

            # Remove from the list of remaining columns
            remaining_columns.remove(j_column)

        # Also need to reset the original index
        df.reset_index(drop=True, inplace=True)

        df_normalized = pd.concat([df[remaining_columns]] + json_normalized, axis=1)

        return df_normalized

    def _rename_columns(self, df):

        df.rename(columns=self._rename_map, inplace=True)

        df.columns = df.columns.str.replace(self._remove_characters, '')
        df.columns = df.columns.str.strip()

        return df

    def _process_columns_old(self, df):

        per_day = df[self._config.ae.timestamp_column_name].dt.to_period("D")

        # Create the per-user, per-day log count
        df["logcount"] = df.groupby([self._config.ae.userid_column_name, per_day]).cumcount()

        # # Create the per-user, per-day location increment count
        # df["locincrement"] = df.groupby([self._config.ae.userid_column_name,
        #                                  per_day])["locationcity"].transform(lambda x: pd.factorize(x)[0] + 1)

        return df

    def _process_single_column(self, ci: ColumnInfo, df: cudf.DataFrame):

        if (ci.input_name not in df.columns):
            # Generate warning?
            # Create empty column
            return cudf.Series(None, dtype=ci.dtype)

        input_col = df[ci.input_name]

        # TODO(MDD): Should we do some processing on converting types if they dont match?

        if (ci.process_column is not None):
            return ci.process_column(df)

        return input_col

    def _process_columns(self, df: cudf.DataFrame):

        output_df = cudf.DataFrame()

        # Iterate over the column info
        for ci in self._column_info:
            output_df[ci.name] = self._process_single_column(ci, df)

        return output_df

    def _get_features(self, df):

        final_columns = self._config.ae.feature_columns + self._additional_columns
        for feature in final_columns:
            if (feature not in df.columns):
                df[feature] = ""

        return df[final_columns]

    def process_features(self, df_in: cudf.DataFrame):
        if (df_in is None):
            return None

        start_time = time.time()
        batch_count = df_in["batch_count"].iloc[0]
        origin_hash = df_in["origin_hash"].iloc[0]

        # Step 1 is to normalize any columns
        df_processed = self._normalize_dataframe(df_in)

        # Step 2 is to process columns
        # df_processed = self._rename_columns(df_processed)
        df_processed = self._process_columns(df_processed)
        # df_processed = self._get_features(df_processed)

        duration = (time.time() - start_time) * 1000.0

        logger.debug("Preprocessed %s data for logs in %s to %s in %s ms",
                     len(df_in),
                     df_processed[self._config.ae.timestamp_column_name].min(),
                     df_processed[self._config.ae.timestamp_column_name].max(),
                     duration)

        self._cache_ids.append(origin_hash)

        cache_location = os.path.join(self._s3_cache_dir, self._cache_path, f"{origin_hash}.parquet")
        if (not os.path.exists(os.path.dirname(cache_location))):
            os.makedirs(os.path.dirname(cache_location), exist_ok=True)

        if (self._generic_only):
            df_processed.to_parquet(cache_location)
            self._df_user_frames = self._df_user_frames.append(
                {
                    "username": "generic_user", "frame_path": cache_location
                }, ignore_index=True)
        else:
            unique_users = df_processed["username"].unique().to_pandas().to_list()
            for user in unique_users:
                self._df_user_frames = self._df_user_frames.append({
                    "username": user, "frame_path": cache_location
                },
                                                                   ignore_index=True)

        # If we've got our full batch processed, push it forward, otherwise return none
        if (len(self._cache_ids) == batch_count):
            user_frames = []
            df_grouped = self._df_user_frames.groupby('username')
            for user_id in df_grouped.groups:
                user_frame_message = UserMessageMeta(df=df_grouped.get_group(user_id), user_id=user_id)
                user_frames.append(user_frame_message)

            return user_frames

        return []

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.process_features), ops.flatten()).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, UserMessageMeta


class DFPPreprocessingStage(SinglePortStage):

    def __init__(self, c: Config, input_schema: DataFrameInputSchema, return_format: str = "data"):
        super().__init__(c)

        self._cache_ids = []
        self._input_schema = input_schema
        self._df_user_frames = pd.DataFrame(columns=("username", "frame_path"))
        self._cache_path = "preprocessing"
        # self._s3_cache_dir = frame_cache_dir
        self._return_format = return_format

    @property
    def name(self) -> str:
        return "dfp-preproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (UserMessageMeta, )

    @staticmethod
    def _create_lonicrement(df):
        slot_list = []
        timeslots = df['time'].unique()
        for slot in timeslots:
            new_df = df[(df['time'] == slot)]
            new_df["locincrement"] = ((new_df['locationcity'].factorize()[0] + 1))
            slot_list.append(new_df)

        return pd.concat(slot_list)

    @staticmethod
    def _create_logcount(df):
        df["logcount"] = df.groupby('time').cumcount()

        return df

    def _normalize_dataframe(self, df):

        json_normalized = []
        remaining_columns = list(df.columns)

        for j_column in self._json_columns:

            if (j_column not in remaining_columns):
                continue

            normalized = pd.json_normalize(df[j_column])

            # Prefix the columns
            normalized.rename(columns={n: f"{j_column}.{n}" for n in normalized.columns}, inplace=True)

            # Reset the index otherwise there is a conflict
            normalized.reset_index(drop=True, inplace=True)

            json_normalized.append(normalized)

            # Remove from the list of remaining columns
            remaining_columns.remove(j_column)

        # Also need to reset the original index
        df.reset_index(drop=True, inplace=True)

        df_normalized = pd.concat([df[remaining_columns]] + json_normalized, axis=1)

        return df_normalized

    def _rename_columns(self, df):

        df.rename(columns=self._rename_map, inplace=True)

        df.columns = df.columns.str.replace(self._remove_characters, '')
        df.columns = df.columns.str.strip()

        return df

    def _process_columns_old(self, df):

        per_day = df[self._config.ae.timestamp_column_name].dt.to_period("D")

        # Create the per-user, per-day log count
        df["logcount"] = df.groupby([self._config.ae.userid_column_name, per_day]).cumcount()

        # # Create the per-user, per-day location increment count
        # df["locincrement"] = df.groupby([self._config.ae.userid_column_name,
        #                                  per_day])["locationcity"].transform(lambda x: pd.factorize(x)[0] + 1)

        return df

    def _process_single_column(self, ci: ColumnInfo, df: cudf.DataFrame):

        if (ci.input_name not in df.columns):
            # Generate warning?
            # Create empty column
            return cudf.Series(None, dtype=ci.dtype)

        input_col = df[ci.input_name]

        # TODO(MDD): Should we do some processing on converting types if they dont match?

        if (ci.process_column is not None):
            return ci.process_column(df)

        return input_col

    def _process_columns(self, df: cudf.DataFrame):

        output_df = cudf.DataFrame()

        # Iterate over the column info
        for ci in self._column_info:
            output_df[ci.name] = self._process_single_column(ci, df)

        return output_df

    def _get_features(self, df):

        final_columns = self._config.ae.feature_columns + self._additional_columns
        for feature in final_columns:
            if (feature not in df.columns):
                df[feature] = ""

        return df[final_columns]

    def process_features(self, message: UserMessageMeta):
        if (message is None):
            return None

        start_time = time.time()

        # Process the columns
        df_processed = process_dataframe(message.df, self._input_schema)

        # Create the multi message
        output_message = UserMessageMeta(df_processed, user_id=message.user_id)

        duration = (time.time() - start_time) * 1000.0

        logger.debug("Preprocessed %s data for logs in %s to %s in %s ms",
                     message.count,
                     message.df[self._config.ae.timestamp_column_name].min(),
                     message.df[self._config.ae.timestamp_column_name].max(),
                     duration)

        return [output_message]

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.process_features), ops.flatten()).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, UserMessageMeta
