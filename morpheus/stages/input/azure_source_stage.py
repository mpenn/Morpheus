# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import glob
import json
import logging
import os
import queue
import typing
from functools import partial

import srf
import numpy as np
import pandas as pd
from srf.core import operators as ops

from morpheus._lib.common import FiberQueue
from morpheus._lib.file_types import FileTypes
from morpheus._lib.file_types import determine_file_type
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import UserMessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.autoencoder_source_stage import AutoencoderSourceStage
from morpheus.utils.producer_consumer_queue import Closed

logger = logging.getLogger(__name__)


class AzureSourceStage(AutoencoderSourceStage):

    @property
    def name(self) -> str:
        return "from-azure"

    def supports_cpp_node(self):
        return False

    @staticmethod
    def change_timestamp(df):
        df['time'] = pd.to_datetime(df['createdDateTime'])
        df['time'] = df['time'].astype(str)
        df['time'] = df['time'].str.split(' ').str[0]
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def change_columns(df):
        df.columns = df.columns.str.replace('[_,.,{,},:]', '')
        df.columns = df.columns.str.strip()
        return df

    @staticmethod
    def create_locincrement(df):
        slot_list = []
        timeslots = df['time'].unique()
        for slot in timeslots:
            new_df = df[df['time'] == slot]
            new_df["locincrement"] = new_df['locationcity'].factorize()[0] + 1
            slot_list.append(new_df)
        if slot_list:
            slot_df = pd.concat(slot_list)
            return slot_df
        df['locincrement'] = np.NaN
        return df

    @staticmethod
    def create_appincrement(df):
        slot_list = []
        timeslots = df['time'].unique()
        for slot in timeslots:
            new_df = df[df['time'] == slot]
            new_df["appincrement"] = new_df['appDisplayName'].factorize()[0] + 1
            slot_list.append(new_df)
        if slot_list:
            slot_df = pd.concat(slot_list)
            return slot_df
        df['locincrement'] = np.NaN
        return df

    @staticmethod
    def create_logcount(df):
        df["logcount"] = df.groupby('time').cumcount()
        return df

    @staticmethod
    def derive_features(df: pd.DataFrame, feature_columns: typing.List[str]):

        df = AzureSourceStage.change_timestamp(df)
        df = AzureSourceStage.change_columns(df) 
        df = AzureSourceStage.create_locincrement(df)
        df = AzureSourceStage.create_appincrement(df)
        df = AzureSourceStage.create_logcount(df)
        
        if (feature_columns is not None):
            df.drop(columns=df.columns.difference(feature_columns), inplace=True)

        return df

    @staticmethod
    def files_to_dfs_per_user(x: typing.List[str],
                              userid_column_name: str,
                              feature_columns: typing.List[str],
                              userid_filter: str = None,
                              repeat_count: int = 1) -> typing.Dict[str, pd.DataFrame]:

        dfs = []
        for file in x:
            df = pd.read_json(file, orient="records")
            df = pd.json_normalize(df['properties'])
            dfs = dfs + AutoencoderSourceStage.repeat_df(df, repeat_count)

        df_per_user = AutoencoderSourceStage.batch_user_split(dfs, userid_column_name, userid_filter)

        return df_per_user
