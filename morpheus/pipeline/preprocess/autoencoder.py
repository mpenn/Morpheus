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
from functools import partial

import cupy as cp
import dill
from dfencoder import AutoEncoder

from morpheus.config import Config
from morpheus.pipeline.messages import InferenceMemoryAE
from morpheus.pipeline.messages import MultiInferenceAEMessage
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.preprocessing import PreprocessBaseStage


class PreprocessAEStage(PreprocessBaseStage):
    """
    Autoencoder usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length

        with open(c.ae.autoencoder_path, 'rb') as in_strm:
            self._autoencoder = dill.load(in_strm)

    @property
    def name(self) -> str:
        return "preprocess-autoencoder"

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, autoencoder: AutoEncoder) -> MultiInferenceAEMessage:
        """
        This function performs pre-processing for autoencoder.

        Parameters
        ----------
        x : morpheus.messages.MultiMessage
            Input rows received from Deserialized stage.
        autoencoder : dfencoder.AutoEncoder
            Autoencoder used to encode model input

        Returns
        -------
        autoencoder_messages.MultiInferenceAutoencoderMessage
            infer_message

        """

        meta_df = x.get_meta()

        data = autoencoder.prepare_df(meta_df)
        input = autoencoder.build_input_tensor(data)
        input = cp.asarray(input.detach())

        count = input.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        memory = InferenceMemoryAE(count=count, input=input, seq_ids=seg_ids)

        infer_message = MultiInferenceAEMessage(meta=x.meta,
                                                mess_offset=x.mess_offset,
                                                mess_count=x.mess_count,
                                                memory=memory,
                                                offset=0,
                                                count=memory.count)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        return partial(PreprocessAEStage.pre_process_batch, fea_len=self._fea_length, autoencoder=self._autoencoder)
