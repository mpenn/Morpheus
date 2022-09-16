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

import logging
import time
import typing

import pandas as pd
import srf
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.column_info import DataFrameInputSchema
from ..utils.column_info import process_dataframe

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPPreprocessingStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 input_schema: DataFrameInputSchema,
                 return_format: str = "data",
                 only_new_batches=False):
        super().__init__(c)

        self._cache_ids = []
        self._input_schema = input_schema
        self._df_user_frames = pd.DataFrame(columns=("username", "frame_path"))
        self._cache_path = "preprocessing"
        self._return_format = return_format
        self._only_new_batches = only_new_batches

    @property
    def name(self) -> str:
        return "dfp-preproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiDFPMessage, )

    def process_features(self, message: MultiDFPMessage):
        if (message is None):
            return None

        start_time = time.time()

        # Process the columns
        df_processed = process_dataframe(message.get_meta_dataframe(), self._input_schema)

        # Apply the new dataframe, only the rows in the offset
        message.set_meta(list(df_processed.columns),
                         df_processed.iloc[message.mess_offset:message.mess_offset + message.mess_count])

        if logger.isEnabledFor(logging.DEBUG):
            duration = (time.time() - start_time) * 1000.0

            logger.debug("Preprocessed %s data for logs in %s to %s in %s ms",
                         message.mess_count,
                         message.get_meta(self._config.ae.timestamp_column_name).min(),
                         message.get_meta(self._config.ae.timestamp_column_name).max(),
                         duration)

        return message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.process_features)).subscribe(sub)

        node = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], node)

        # node.launch_options.pe_count = self._config.num_threads

        return node, MultiDFPMessage