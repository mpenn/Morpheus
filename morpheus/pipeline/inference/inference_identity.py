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

import asyncio
import queue
import typing

import cupy as cp
from tornado.ioloop import IOLoop

from morpheus.config import Config
from morpheus.pipeline.inference.inference_stage import InferenceStage, InferenceWorker
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import ResponseMemoryProbs


# This class is exclusively run in the worker thread. Separating the classes helps keeps the threads separate
class IdentityInferenceWorker(InferenceWorker):
    def __init__(self, inf_queue: ProducerConsumerQueue, c: Config):
        super().__init__(inf_queue)

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.feature_length

    def process(self, batch: MultiInferenceMessage, fut: asyncio.Future):
        def tmp(b: MultiInferenceMessage, f):

            f.set_result(
                ResponseMemoryProbs(
                    count=b.count,
                    probs=cp.zeros((b.count, self._seq_length), dtype=cp.float32),
                ))

        self._loop.add_callback(tmp, batch, fut)


class IdentityInferenceStage(InferenceStage):
    def __init__(self, c: Config):
        super().__init__(c)

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:
        return IdentityInferenceWorker(inf_queue=inf_queue, c=Config.get())
