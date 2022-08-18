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
import typing

import numpy as np
import pandas as pd
import srf
import torch
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .dfp_autoencoder import DFPAutoEncoder
from .user_model_manager import UserModelManager

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPTraining(SinglePortStage):

    def __init__(self, c: Config):
        super().__init__(c)

        self._user_models: typing.Dict[str, UserModelManager] = {}

    @property
    def name(self) -> str:
        return "dfp-training"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def on_data(self, message: UserMessageMeta):
        if (message is None or message.df.empty):
            return None

        user = message.user_id

        model_manager = UserModelManager(self._config,
                                         user_id=user,
                                         save_model=False,
                                         epochs=30,
                                         min_history=300,
                                         max_history=-1,
                                         seed=42,
                                         model_class=DFPAutoEncoder)

        model = model_manager.train(message.df)

        output_message = MultiAEMessage(message, mess_offset=0, mess_count=message.count, model=model)

        return output_message

        # cached_batch_frames = message.df["frame_path"].to_list()
        # self._user_models[user] = UserModelManager(self._config,
        #                                            user_id=user,
        #                                            save_model=False,
        #                                            epochs=30,
        #                                            min_history=300,
        #                                            max_history=-1,
        #                                            seed=42,
        #                                            batch_files=cached_batch_frames,
        #                                            model_class=DFPAutoEncoder)

        # if (user == self._config.ae.fallback_username):
        #     model, sample_frame = self._user_models[user].train_from_batch()
        # else:
        #     model, sample_frame = self._user_models[user].train_from_batch(filter_func=
        #                                                                    lambda df: df[df['username'] == user])

        # # Can return None if there wasn't enough history
        # if (model):
        #     user_message = UserMessageMeta(df=sample_frame, user_id=user)
        #     return MultiAEMessage(user_message, mess_offset=0, mess_count=user_message.count, model=model)
        # else:
        #     return None

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, MultiAEMessage
