# Copyright (c) 2021, NVIDIA CORPORATION.
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
from functools import reduce

import cupy as cp
from morpheus.config import Config
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.messages import MultiResponseMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.pipeline import StreamPair


class AddClassificationsStage(SinglePortStage):
    def __init__(self, c: Config, threshold: float = 0.5):
        super().__init__(c)

        self._threshold = threshold

    @property
    def name(self) -> str:
        return "add-class"

    def accepted_types(self) -> typing.Tuple:
        return (MultiResponseMessage, )

    def _add_labels(self, x: MultiResponseMessage):

        probs = x.probs[:,1]
        preds = (probs > self._threshold).astype(cp.bool).get()

        x.set_meta("pred", preds.tolist())
        x.set_meta("score", probs.tolist())

        # Return list of strs to write out
        return x

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Convert the messages to rows of strings
        stream = stream.async_map(self._add_labels, executor=self._pipeline.thread_pool)

        # Return input unchanged
        return stream, MultiResponseProbsMessage

