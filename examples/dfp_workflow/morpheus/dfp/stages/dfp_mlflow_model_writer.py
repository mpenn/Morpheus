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
import os
import typing

import mlflow
import numpy as np
import srf
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.protos.databricks_pb2 import ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .dfp_autoencoder import DFPAutoEncoder

# Setup conda environment
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': ['python={}'.format('3.8'), 'pip'],
    'pip': ['mlflow', 'dfencoder'],
    'name': 'mlflow-env'
}

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPMLFlowModelWriterStage(SinglePortStage):

    def __init__(self, c: Config, model_name_formatter: str, experiment_name: str):
        super().__init__(c)

        self._model_name_formatter = model_name_formatter
        self._experiment_name = experiment_name

        self._batch_size = 10
        self._batch_cache = []

    @property
    def name(self) -> str:
        return "dfp-mlflow-model-writer"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiAEMessage, )

    def on_data(self, message: MultiAEMessage):

        user = message.meta.user_id
        model: DFPAutoEncoder = message.model

        model_path = "dfencoder"
        reg_model_name = self._model_name_formatter.format(user_id=user)

        # Write to ML Flow
        try:
            mlflow.end_run()
            experiment_name = os.path.join(self._experiment_name, reg_model_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if (not experiment):
                # print(f"Failed to get experiment: /heimdall-dsp/{reg_model_name}")
                # print(f"Creating...")
                mlflow.create_experiment(experiment_name)
                experiment = mlflow.get_experiment_by_name(experiment_name)

            with mlflow.start_run(run_name="Duo autoencoder model training run",
                                  experiment_id=experiment.experiment_id) as run:

                model_path = f"{model_path}-{run.info.run_uuid}"
                # print(f"Model path: {model_path}")
                mlflow.log_param("Algorithm", "Denosing Autoencoder")
                mlflow.log_param("Epochs", 30)
                mlflow.log_param("Learning rate", 0.001)
                mlflow.log_param("Batch size", 21)

                # Temp values
                loss = [[1, 2]]
                loss_val = np.mean([(x[0]) for x in loss])
                loss_val_std = np.std([x[0] for x in loss])
                swapped_loss = np.mean([x[1] for x in loss])

                # Get the val loss and std dev
                loss_val = model.val_loss_mean
                loss_val_std = model.val_loss_std

                mlflow.log_metric("Mean validation loss", float(loss_val))
                mlflow.log_metric("Mean swapped loss", float(swapped_loss))
                mlflow.log_metric("Mean loss std", float(loss_val_std))

                # Use the prepare_df function to setup the direct inputs to the model
                sample_input = model.prepare_df(message.get_meta())

                # TODO(MDD) this should work with sample_input
                model_sig = infer_signature(message.get_meta(), model.get_anomaly_score(sample_input))

                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=model_path,
                    # registered_model_name=reg_model_name, -- We register this below and use it as a check
                    conda_env=conda_env,
                    signature=model_sig,
                )

                # print("Model info: ")
                # print(model_info.model_uri)
                # print(model_info.artifact_path)

                client = MlflowClient()

                # First ensure a registered model has been created
                try:
                    create_model_response = client.create_registered_model(reg_model_name)
                    # print("Successfully registered model '%s'." % create_model_response.name, flush=True)
                except MlflowException as e:
                    if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                        pass
                    else:
                        raise e

                model_src = RunsArtifactRepository.get_underlying_uri(model_info.model_uri)
                # print(f"Model src: {model_src}")

                tags = {
                    "start": message.get_meta(self._config.ae.timestamp_column_name).min(),
                    "end": message.get_meta(self._config.ae.timestamp_column_name).max(),
                    "count": message.get_meta(self._config.ae.timestamp_column_name).count()
                }

                # Now create the model version
                mv = client.create_model_version(name=reg_model_name,
                                                 source=model_src,
                                                 run_id=run.info.run_id,
                                                 tags=tags)

                logger.debug("ML Flow model upload complete. User: %s, Version: %s", user, mv.version)

        except Exception as e:
            logger.exception("Error trying to upload ML Flow model")
        mlflow.end_run()

        # print(f"Finished registering MLflow model for user: {user}", flush=True)
        return message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, MultiAEMessage
