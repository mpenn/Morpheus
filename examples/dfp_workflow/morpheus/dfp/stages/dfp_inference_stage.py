import logging
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


class DFPInferenceStage(SinglePortStage):

    def __init__(self, c: Config):
        super().__init__(c)

        self._client = MlflowClient()
        self._fallback_user = "generic_user"

        self._model_cache = {}
        self._model_cache_size_max = 10

    @property
    def name(self) -> str:
        return "dfp-inference"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta, )

    def get_model(self, user: str) -> typing.Tuple[DFPAutoEncoder, str]:
        now = datetime.now()
        reg_model_name = f"autoencoder-HAMMAH-duo-{user}"

        if (reg_model_name not in self._model_cache
                or ((now - self._model_cache[reg_model_name]["last_checked"]).seconds >
                    600)):  # 10 minute timeout between checks
            try:
                latest_versions = self._client.get_latest_versions(reg_model_name)

                # Default to the first returned one
                latest_model_version = latest_versions[0]

                if (len(latest_versions) > 1):
                    logger.warning(
                        "Multiple models in different stages detected. Defaulting to first returned. Version: %s, Stage: %s",
                        latest_model_version.version,
                        latest_model_version.current_stage)

                model_uri = self._client.get_model_version_download_uri(reg_model_name, latest_model_version.version)

                loaded_model: DFPAutoEncoder = mlflow.pytorch.load_model(model_uri=model_uri)
                self._model_cache[reg_model_name] = {
                    "model": loaded_model,
                    "model_uri": model_uri,
                    "last_checked": now,
                    "last_used": now,
                    "latest_version": latest_model_version
                }

            except MlflowException as e:
                return None, ""

        cache_entry = self._model_cache[reg_model_name]
        cache_entry["last_used"] = now

        if (len(self._model_cache) > self._model_cache_size_max):
            time_sorted = sorted([(k, v) for k, v in self._model_cache.items()], key=lambda x: x[1]["last_used"])
            to_delete = time_sorted[0][0]
            self._model_cache.pop(to_delete)

        return cache_entry["model"], cache_entry["model_uri"]

    def on_data(self, message: UserMessageMeta):
        if (not message or message.df.empty):
            return None

        start_time = time.time()

        df_user = message.df
        user = message.user_id

        #logger.debug("Inference got user: %s", user)

        try:
            loaded_model, model_uri = self.get_model(user)
            if (loaded_model is None):
                loaded_model, model_uri = self.get_model(self._fallback_user)

            if (loaded_model is None):
                raise RuntimeError("Could not find model for user {}".format(user))

        except Exception as e:  # TODO
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

        output_message.set_meta('model_version', model_uri)

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

        # node.launch_options.pe_count = self._config.num_threads

        return node, MultiAEMessage
