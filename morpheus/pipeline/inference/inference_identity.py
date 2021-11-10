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

import cupy as cp

from morpheus.config import Config
from morpheus.pipeline.inference.inference_stage import InferenceStage
from morpheus.pipeline.inference.inference_stage import InferenceWorker
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import ResponseMemoryProbs
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


# This class is exclusively run in the worker thread. Separating the classes helps keeps the threads separate
class IdentityInferenceWorker(InferenceWorker):
    def __init__(self, inf_queue: ProducerConsumerQueue, c: Config):
        super().__init__(inf_queue)

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.feature_length

    def calc_output_dims(self, x: MultiInferenceMessage) -> typing.Tuple:
        return (x.count, self._seq_length)

    def process(self, batch: MultiInferenceMessage, cb: typing.Callable[[ResponseMemoryProbs], None]):
        def tmp(b: MultiInferenceMessage, f):

            f(ResponseMemoryProbs(
                count=b.count,
                probs=cp.zeros((b.count, self._seq_length), dtype=cp.float32),
            ))

        # self._loop.add_callback(tmp, batch, fut)
        # Call directly instead of enqueing
        tmp(batch, cb)


class IdentityInferenceStage(InferenceStage):
    def __init__(self, c: Config):
        super().__init__(c)

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:
        return IdentityInferenceWorker(inf_queue=inf_queue, c=Config.get())
