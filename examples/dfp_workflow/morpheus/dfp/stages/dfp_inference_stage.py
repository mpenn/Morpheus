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

import dataclasses
import logging
import threading
import time
import typing
from datetime import datetime

import mlflow
import numpy as np
import srf
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.protos.databricks_pb2 import ErrorCode
from mlflow.tracking.client import MlflowClient

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .dfp_autoencoder import DFPAutoEncoder

logger = logging.getLogger("morpheus.{}".format(__name__))


class ModelCache:

    def __init__(self, reg_model_name: str, model_uri: str) -> None:

        self._reg_model_name = reg_model_name
        self._model_uri = model_uri

        self._last_checked: datetime = datetime.now()
        self._last_used: datetime = self._last_checked

        self._lock = threading.Lock()
        self._model: DFPAutoEncoder = None

    @property
    def model_uri(self):
        return self._model_uri

    @property
    def last_used(self):
        return self._last_used

    @property
    def last_checked(self):
        return self._last_checked

    def load_model(self, client):

        now = datetime.now()

        with self._lock:

            if (self._model is None):

                # Cache miss. Release the lock while we check
                try:
                    self._model = mlflow.pytorch.load_model(model_uri=self._model_uri)

                except MlflowException:
                    logger.error("Error downloading model for URI: %s", self._model_uri, exc_info=True)
                    raise

            # Update the last time this was used
            self._last_used = now

            return self._model


class UserModelMap:

    def __init__(self, manager: "ModelManager", user_id: str, fallback_user_ids: typing.List[str]):

        self._manager = manager
        self._user_id = user_id
        self._fallback_user_ids = fallback_user_ids
        self._reg_model_name = manager.user_id_to_model(user_id)
        self._last_checked = None

        self._lock = threading.Lock()
        self._child_user_model_cache: UserModelMap = None

    def load_model(self, client):

        now = datetime.now()

        # Lock to prevent additional access
        with self._lock:

            # Check if we have checked before or if we need to check again
            if (self._last_checked is None or (now - self._last_checked).seconds < self._manager.cache_timeout_sec):

                # Save the last checked time
                self._last_checked = now

                # Try to load from the manager
                model_cache = self._manager.load_model_cache(client=client, reg_model_name=self._reg_model_name)

                # If we have a hit, there is nothing else to do
                if (model_cache is None):
                    # Our model does not exist, use fallback
                    self._child_user_model_cache = self._manager.load_user_model_cache(
                        self._fallback_user_ids[0], fallback_user_ids=self._fallback_user_ids[1:])

            # See if we have a child cache and use that
            if (self._child_user_model_cache is not None):
                return self._child_user_model_cache.load_model(client=client)

            # Otherwise load the model
            model_cache = self._manager.load_model_cache(client=client, reg_model_name=self._reg_model_name)

            if (model_cache is None):
                raise RuntimeError("Model was found but now no longer exists. Model: {}".format(self._reg_model_name))

            return model_cache


class ModelManager:

    def __init__(self, model_format_str: str = "autoencoder-HAMMAH-duo-{user}") -> None:
        self._model_format_str = model_format_str

        self._user_model_cache: typing.Dict[str, UserModelMap] = {}

        self._model_cache: typing.Dict[str, ModelCache] = {}
        self._model_cache_size_max = 10

        self._cache_timeout_sec = 600

        self._user_model_cache_lock = threading.Lock()
        self._model_cache_lock = threading.Lock()

    @property
    def cache_timeout_sec(self):
        return self._cache_timeout_sec

    def user_id_to_model(self, user_id: str):
        return self._model_format_str.format(user=user_id)

    def load_user_model(self, client, user_id: str, fallback_user_ids: typing.List[str] = []) -> ModelCache:

        # First get the UserModel
        user_model_cache = self.load_user_model_cache(user_id=user_id, fallback_user_ids=fallback_user_ids)

        return user_model_cache.load_model(client=client)

    def load_model_cache(self, client, reg_model_name: str) -> ModelCache:

        now = datetime.now()

        with self._model_cache_lock:

            model_cache = self._model_cache.get(reg_model_name, None)

            # Make sure it hasnt been too long since we checked
            if (model_cache is not None and (now - model_cache.last_checked).seconds < self._cache_timeout_sec):

                return model_cache

            # Cache miss. Try to check for a model
            try:
                latest_versions = client.get_latest_versions(reg_model_name)

                # Default to the first returned one
                latest_model_version = latest_versions[0]

                if (len(latest_versions) > 1):
                    logger.warning(("Multiple models in different stages detected. "
                                    "Defaulting to first returned. Version: %s, Stage: %s"),
                                   latest_model_version.version,
                                   latest_model_version.current_stage)

                model_cache = ModelCache(reg_model_name=reg_model_name, model_uri=latest_model_version.source)

            except MlflowException:
                # No user found
                return None

            # Save the cache
            self._model_cache[reg_model_name] = model_cache

            # Check if we need to push out a cache entry
            if (len(self._model_cache) > self._model_cache_size_max):
                time_sorted = sorted([(k, v) for k, v in self._model_cache.items()], key=lambda x: x[1].last_used)
                to_delete = time_sorted[0][0]
                self._model_cache.pop(to_delete)

            return model_cache

    def load_user_model_cache(self, user_id: str, fallback_user_ids: typing.List[str] = []) -> UserModelMap:
        with self._user_model_cache_lock:

            if (user_id not in self._user_model_cache):
                self._user_model_cache[user_id] = UserModelMap(manager=self,
                                                               user_id=user_id,
                                                               fallback_user_ids=fallback_user_ids)

            return self._user_model_cache[user_id]


class DFPInferenceStage(SinglePortStage):

    def __init__(self, c: Config, model_format_str: str = "autoencoder-HAMMAH-duo-{user}"):
        super().__init__(c)

        self._client = MlflowClient()
        self._fallback_user = self._config.ae.fallback_username
        self._model_format_str = model_format_str

        self._users_to_models: typing.Dict[str, UserModelMap] = {}

        self._model_cache: typing.Dict[str, ModelCache] = {}
        self._model_cache_size_max = 10

        self._cache_timeout_sec = 600

        self._model_manager = ModelManager(model_format_str=model_format_str)

    @property
    def name(self) -> str:
        return "dfp-inference"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta, )

    # def _get_model_map_for_user(self, user_id: str) -> UserModelMap:
    #     # First check the user to model cache
    #     model_map = self._users_to_models.get(user_id, None)

    #     # Make sure it hasnt been too long since we checked
    #     if (model_map is not None and (datetime.now() - model_map.last_checked).seconds < self._cache_timeout_sec):
    #         return model_map

    #     # Cache miss, get the registered model name
    #     reg_model_name = self._model_format_str.format(user=user_id)

    #     try:
    #         latest_versions = self._client.get_latest_versions(reg_model_name)

    #         # Default to the first returned one
    #         latest_model_version = latest_versions[0]

    #         if (len(latest_versions) > 1):
    #             logger.warning(
    #                 "Multiple models in different stages detected. Defaulting to first returned. Version: %s, Stage: %s",
    #                 latest_model_version.version,
    #                 latest_model_version.current_stage)

    #         # Got a version, create new cache entry
    #         model_map = UserModelMap(user_id=user_id,
    #                                  reg_model_name=reg_model_name,
    #                                  reg_model_version=latest_model_version.version,
    #                                  reg_model_uri=latest_model_version.source,
    #                                  last_checked=datetime.now())

    #         self._users_to_models[user_id] = model_map

    #         return model_map
    #     except MlflowException:
    #         # No user found
    #         return None

    # def _get_model_for_user(self, user_id: str, fallback_ids: typing.List[str] = []) -> UserModelMap:

    #     for uid in [user_id] + fallback_ids:
    #         model_map = self._get_model_map_for_user(user_id=uid)

    #         if (model_map is not None):
    #             return model_map

    #     raise RuntimeError("Could not find model for user or fallbacks: {}".format(user_ids))

    # def _get_model_from_map(self, model_map: UserModelMap) -> ModelCache:
    #     now = datetime.now()

    #     # First check the user to model cache
    #     model_cache = self._model_cache.get(model_map.reg_model_name, None)

    #     # Make sure it hasnt been too long since we checked
    #     if (model_cache is not None and (now - model_cache.last_checked).seconds < self._cache_timeout_sec):

    #         # Update the last used
    #         model_cache.last_used = now

    #         return model_cache

    #     # Cache miss
    #     try:
    #         loaded_model: DFPAutoEncoder = mlflow.pytorch.load_model(model_uri=model_map.reg_model_uri)

    #         model_cache = ModelCache(model=loaded_model,
    #                                  model_uri=model_map.reg_model_uri,
    #                                  last_checked=now,
    #                                  last_used=now,
    #                                  latest_version=model_map.reg_model_version)

    #         self._model_cache[model_map.reg_model_name] = model_cache

    #         # Check if we need to push out a cache entry
    #         if (len(self._model_cache) > self._model_cache_size_max):
    #             time_sorted = sorted([(k, v) for k, v in self._model_cache.items()], key=lambda x: x[1].last_used)
    #             to_delete = time_sorted[0][0]
    #             self._model_cache.pop(to_delete)

    #         return model_cache

    #     except MlflowException:
    #         logger.error("Error while downloading model: %s, version: %s",
    #                      model_map.reg_model_name,
    #                      model_map.reg_model_version,
    #                      exc_info=True)
    #         raise

    def get_model(self, user: str) -> ModelCache:

        return self._model_manager.load_user_model(self._client,
                                                   user_id=user,
                                                   fallback_user_ids=[self._config.ae.fallback_username])

        # # First get the model map
        # model_map = self._get_model_for_user([user, self._config.ae.fallback_username])

        # # Now get the actual model
        # return self._get_model_from_map(model_map=model_map)

    def on_data(self, message: UserMessageMeta):
        if (not message or message.df.empty):
            return None

        start_time = time.time()

        df_user = message.df
        user = message.user_id

        try:
            model_cache = self.get_model(user)

            if (model_cache is None):
                raise RuntimeError("Could not find model for user {}".format(user))

            loaded_model = model_cache.load_model(self._client)

        except Exception:  # TODO
            logger.exception("Error trying to get model")
            return None

        post_model_time = time.time()

        anomaly_score = loaded_model.get_anomaly_score(df_user)
        # anomaly_score = np.zeros((message.count, 1), dtype=float)
        # loaded_model.val_loss_mean = 1
        # loaded_model.val_loss_std = 0.2

        # Create an output message to allow setting meta
        output_message = MultiAEMessage(message, mess_offset=0, mess_count=message.count, model=loaded_model)

        output_message.set_meta("anomaly_score", anomaly_score)

        output_message.set_meta('model_version', model_cache.model_uri)

        load_model_duration = (post_model_time - start_time) * 1000.0
        get_anomaly_duration = (time.time() - post_model_time) * 1000.0

        logger.debug("Completed inference for user %s. Model load: %s ms, Model infer: %s ms. Start: %s, End: %s",
                     user,
                     load_model_duration,
                     get_anomaly_duration,
                     df_user[self._config.ae.timestamp_column_name].min(),
                     df_user[self._config.ae.timestamp_column_name].max())

        return output_message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self.on_data)
        builder.make_edge(input_stream[0], node)

        node.launch_options.pe_count = self._config.num_threads

        return node, MultiAEMessage
