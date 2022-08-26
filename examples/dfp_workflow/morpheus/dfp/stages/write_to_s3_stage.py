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

import typing

import srf

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


class WriteToS3Stage(SinglePortStage):
    """
    This class writes messages to an s3 bucket.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    bucket: str
        Name of the s3 bucket to write to.

    """

    def __init__(self, c: Config, s3_writer):
        super().__init__(c)

        self._s3_writer = s3_writer

    @property
    def name(self) -> str:
        return "to-s3-bucket"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        node = builder.make_node(self.unique_name, self._s3_writer)
        builder.make_edge(stream, node)

        stream = node

        # Return input unchanged to allow passthrough
        return stream, input_stream[1]