# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


class RecipientFeaturesStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "recipient-features"

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MessageMeta) -> MessageMeta:
        # Get the DataFrame from the incoming message
        df = message.df

        df['to_count'] = df['To'].str.count('@')
        df['bcc_count'] = df['BCC'].str.count('@')
        df['cc_count'] = df['CC'].str.count('@')
        df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

        # Attach features to string data
        df['data'] = (df['to_count'].astype(str) + '[SEP]' + df['bcc_count'].astype(str) + '[SEP]' +
                      df['cc_count'].astype(str) + '[SEP]' + df['total_recipients'].astype(str) + '[SEP]' +
                      df['Message'])

        # Return the message for the next stage
        return message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self.on_data)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
