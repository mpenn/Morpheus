"""
The ``fil`` module provides APIs for logging and loading Forest Inference Library (FIL)
models in the MLflow Model format. This module exports MLflow Models with the following 
flavors:

FIL format
    Forest models trained by machine learning frameworks like XGBoost, LightGBM, 
    Scikit-Learn, and cuML.

"""
import os
import shutil
import sys
import yaml
import numpy as np

import pandas as pd

from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.utils.annotations import experimental
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

import fil_flavor

FLAVOR_NAME = "fil"


@experimental
def save_model(
    fil_model_path,
    path,
    mlflow_model=None,
):
    """
    Save an FIL model to a path on the local file system.

    :param fil_model_path: File path to FIL model to be saved.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    """

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path), error_code=RESOURCE_ALREADY_EXISTS
        )
    os.makedirs(path)
    model_data_subpath = os.path.basename(fil_model_path)
    model_data_path = os.path.join(path, model_data_subpath)

    # Save FIL model
    shutil.copy(fil_model_path, model_data_path)

    mlflow_model.add_flavor(FLAVOR_NAME, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


@experimental
def log_model(
    fil_model_path,
    artifact_path,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """
    Log an FIL model as an MLflow artifact for the current run.

    :param fil_model_path: File path to FIL model.
    :param artifact_path: Run-relative artifact path.
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    """
    Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        fil_model_path=fil_model_path,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
    )