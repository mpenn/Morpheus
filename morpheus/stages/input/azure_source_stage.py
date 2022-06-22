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


class AzureSourceStage(AutoencoderSourceStage):

    @property
    def name(self) -> str:
        return "from-azure"

    def supports_cpp_node(self):
        return False

    @staticmethod
    def read_file(filename: str, file_type: FileTypes) -> pd.DataFrame:
        """
        Reads a file into a dataframe.

        Parameters
        ----------
        filename : str
            Path to a file to read.
        file_type : `morpheus._lib.file_types.FileTypes`
            What type of file to read. Leave as Auto to auto detect based on the file extension.

        Returns
        -------
        pandas.DataFrame
            The parsed dataframe.

        Raises
        ------
        RuntimeError
            If an unsupported file type is detected.
        """

        df = read_file_to_df(filename, file_type, df_type="pandas", parser_kwargs={"lines": True})

        df = df["_raw"].apply(json.loads).apply(pd.Series)[["location","appDisplayName","userPrincipalName","createdDateTime","riskEventTypes_v2", "status","clientAppUsed","deviceDetail"]]
        df = df.loc[:,~df.columns.duplicated()]
        location_df = pd.json_normalize(df['location'])[["city","state","countryOrRegion"]]
        status_df = pd.json_normalize(df['status'])[["failureReason"]]
        device_df = pd.json_normalize(df["deviceDetail"])[["displayName","browser","operatingSystem"]]
        df = pd.concat([df, location_df, status_df, device_df], axis=1)
        df = df.rename(columns={"countryorRegion": "locationcountryOrRegion" ,"state": "locationstate","city": "locationcity","createdDateTime":"time","displayName":"deviceDetaildisplayName","browser":"deviceDetailbrowser","operatingSystem":"deviceDetailoperatingSystem","failureReason":"statusfailureReason"})

        return df

    @staticmethod
    def derive_features(df: pd.DataFrame, feature_columns: typing.List[str]):
        
        df.columns=df.columns.str.replace('[_,.,{,},:]','')
        df['event_ymd']=df['time'].str.split('T').str[0]
        df.columns = df.columns.str.strip()

        datelist=list(set(list(df["event_ymd"])))
        datelist.sort()
        newloclist=[]
        newapplist=[]

        for dt in (datelist):
            d=(df.index[df['event_ymd'] == dt].tolist())
            df2=df.loc[d]
            df2=df2.fillna("nan")
            df2["locincrement"]=((df2['locationcity'].factorize()[0] + 1))
            newloclist.append(df2["locincrement"])
        if len(newloclist)!=0:
            df3 = pd.concat(newloclist)
            df=pd.concat([df,df3],axis=1)

        for dt in (datelist):
            d=(df.index[df['event_ymd'] == dt].tolist())
            df2=df.loc[d]
            df2=df2.fillna("nan")
            df2["appincrement"]=((df2['appDisplayName'].factorize()[0] + 1))
            newapplist.append(df2["appincrement"])
        if len(newapplist)!=0:
            df3 = pd.concat(newapplist)
            df = pd.concat([df,df3],axis=1)

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
            df = AzureSourceStage.read_file(file, FileTypes.JSON)
            dfs = dfs + AutoencoderSourceStage.repeat_df(df, repeat_count)

        df_per_user = AutoencoderSourceStage.batch_user_split(dfs, userid_column_name, userid_filter)

        return df_per_user
