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
import logging
import os
import queue
import typing
from functools import partial

import neo
import numpy as np
import pandas as pd
from neo.core import operators as ops

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


class DuoSourceStage(AutoencoderSourceStage):

    @property
    def name(self) -> str:
        return "from-duo"

    @staticmethod
    def derive_features(df: pd.DataFrame, feature_columns: typing.List[str]):

        df.columns=df.columns.str.replace('[_,.,{,},:]','')
        df["event_dt"] = pd.to_datetime(df['timestamp'],unit='s')
        df['event_ymd'] = df['event_dt'].astype(str).str.split(' ').str[0]

        datelist=list(set(list(df["event_ymd"])))
        datelist.sort()
        newloclist=[]
        for dt in (datelist):
            d=(df.index[df['event_ymd'] == dt].tolist())
            df2=df.loc[d]
            df2=df2.fillna("nan")
            df2["locincrement"]=((df2['locationcity'].factorize()[0] + 1))
            newloclist.append(df2["locincrement"])
        if len(newloclist)!=0:
            df3 = pd.concat(newloclist)
            df=pd.concat([df,df3],axis=1)

        df["logcount"]=df.groupby('event_ymd').cumcount()

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
            df = read_file_to_df(file, FileTypes.CSV, df_type="pandas")
            dfs = dfs + AutoencoderSourceStage.repeat_df(df, repeat_count)

        df_per_user = AutoencoderSourceStage.batch_user_split(dfs, userid_column_name, userid_filter)

        return df_per_user
