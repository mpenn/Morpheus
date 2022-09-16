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

import functools
import logging
import os
import typing
from datetime import datetime
from functools import partial

import click
import mlflow
from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage
from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
from dfp.stages.dfp_inference_stage import DFPInferenceStage
from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
from dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage
from dfp.stages.dfp_preprocessing_stage import DFPPreprocessingStage
from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage
from dfp.stages.dfp_split_users_stage import DFPSplitUsersStage
from dfp.stages.dfp_training import DFPTraining
from dfp.stages.multi_file_source import MultiFileSource
from dfp.utils.column_info import BoolColumn
from dfp.utils.column_info import ColumnInfo
from dfp.utils.column_info import CustomColumn
from dfp.utils.column_info import DataFrameInputSchema
from dfp.utils.column_info import DateTimeColumn
from dfp.utils.column_info import IncrementColumn
from dfp.utils.column_info import RenameColumn
from dfp.utils.column_info import StringCatColumn
from dfp.utils.column_info import create_increment_col
from dfp.utils.file_utils import date_extractor
from dfp.utils.file_utils import iso_date_regex

from morpheus._lib.file_types import FileTypes
from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import load_labels_file
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logger import configure_logging
from morpheus.utils.logger import get_log_levels
from morpheus.utils.logger import parse_log_level


@click.command()
@click.option(
    "--train_users",
    type=click.Choice(["all", "generic", "individual", "none"], case_sensitive=False),
    help="Indicates whether or not to train per user or a generic model for all users",
)
@click.option(
    "--skip_user",
    multiple=True,
    type=str,
    help="User IDs to skip. Mutually exclusive with only_user",
)
@click.option(
    "--only_user",
    multiple=True,
    type=str,
    help="Only users specified by this option will be included. Mutually exclusive with skip_user",
)
@click.option(
    "--duration",
    type=str,
    default="60d",
    help="The duration to run starting from now",
)
@click.option(
    "--cache_dir",
    type=str,
    default="./.cache/dfp",
    show_envvar=True,
    help="The location to cache data such as S3 downloads and pre-processed data",
)
@click.option("--log_level",
              default=logging.getLevelName(Config().log_level),
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              help="Specify the logging level to use.")
@click.option("--sample_rate_s",
              type=int,
              default=0,
              show_envvar=True,
              help="Minimum time step, in milliseconds, between object logs.")
@click.option(
    "--input_file",
    "-f",
    type=str,
    multiple=True,
    help=("List of files to process. Can specificy multiple arguments for multiple files. "
          "Also accepts glob (*) wildcards and schema prefixes such as `s3://`. "
          "For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. "
          "See fsspec documentation for list of possible options."),
)
@click.option('--tracking_uri',
              type=str,
              default="http://localhost:5000",
              help=("The ML Flow tracking URI to connect to the tracking backend. If not speficied, MF Flow will use "
                    "'file:///mlruns' relative to the current directory"))
def run_pipeline(train_users,
                 skip_user: typing.Tuple[str],
                 only_user: typing.Tuple[str],
                 duration,
                 cache_dir,
                 log_level,
                 sample_rate_s,
                 **kwargs):
    # To include the generic, we must be training all or generic
    include_generic = train_users == "all" or train_users == "generic"

    # To include individual, we must be either training or inferring
    include_individual = train_users != "generic"

    # None indicates we arent training anything
    is_training = train_users != "none"

    skip_users = list(skip_user)
    only_users = list(only_user)

    # Enable the Morpheus logger
    configure_logging(log_level=log_level)

    if (len(skip_users) > 0 and len(only_users) > 0):
        logging.error("Option --skip_user and --only_user are mutually exclusive. Exiting")

    logger = logging.getLogger("morpheus.{}".format(__name__))

    logger.info("Running training pipeline with the following options: ")
    logger.info("Train generic_user: %s", include_generic)
    logger.info("Skipping users: %s", skip_users)
    logger.info("Duration: %s", duration)
    logger.info("Cache Dir: %s", cache_dir)

    if ("tracking_uri" in kwargs):
        # Initialize ML Flow
        mlflow.set_tracking_uri(kwargs["tracking_uri"])
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    config = Config()

    CppConfig.set_should_use_cpp(False)

    config.num_threads = os.cpu_count()

    config.ae = ConfigAutoEncoder()

    config.ae.feature_columns = load_labels_file(get_package_relative_file("data/columns_ae_duo.txt"))
    config.ae.userid_column_name = "username"
    config.ae.timestamp_column_name = "timestamp"

    # Specify the column names to ensure all data is uniform
    source_column_info = [
        DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name="timestamp"),
        RenameColumn(name=config.ae.userid_column_name, dtype=str, input_name="user.name"),
        RenameColumn(name="accessdevicebrowser", dtype=str, input_name="access_device.browser"),
        RenameColumn(name="accessdeviceos", dtype=str, input_name="access_device.os"),
        StringCatColumn(name="location",
                        dtype=str,
                        input_columns=[
                            "access_device.location.city",
                            "access_device.location.state",
                            "access_device.location.country"
                        ],
                        sep=", "),
        RenameColumn(name="authdevicename", dtype=str, input_name="auth_device.name"),
        BoolColumn(name="result",
                   dtype=bool,
                   input_name="result",
                   true_values=["success", "SUCCESS"],
                   false_values=["denied", "DENIED", "FRAUD"]),
        ColumnInfo(name="reason", dtype=str),
        # CustomColumn(name="user.groups", dtype=str, process_column_fn=partial(column_listjoin, col_name="user.groups"))
    ]

    source_schema = DataFrameInputSchema(json_columns=["access_device", "application", "auth_device", "user"],
                                         column_info=source_column_info)

    # Preprocessing schema
    preprocess_column_info = [
        ColumnInfo(name=config.ae.timestamp_column_name, dtype=datetime),
        ColumnInfo(name=config.ae.userid_column_name, dtype=str),
        ColumnInfo(name="accessdevicebrowser", dtype=str),
        ColumnInfo(name="accessdeviceos", dtype=str),
        ColumnInfo(name="authdevicename", dtype=str),
        ColumnInfo(name="result", dtype=bool),
        ColumnInfo(name="reason", dtype=str),
        # Derived columns
        IncrementColumn(name="logcount",
                        dtype=int,
                        input_name=config.ae.timestamp_column_name,
                        groupby_column=config.ae.userid_column_name),
        CustomColumn(name="locincrement",
                     dtype=int,
                     process_column_fn=partial(create_increment_col, column_name="location")),
    ]

    preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    pipeline.set_source(MultiFileSource(config, filenames=list(kwargs["input_file"])))

    # Batch files into buckets by time. Use the default ISO date extractor from the filename
    pipeline.add_stage(
        DFPFileBatcherStage(config,
                            period="D",
                            sampling_rate_s=sample_rate_s,
                            date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex)))

    # Output is S3 Buckets. Convert to DataFrames. This caches downloaded S3 data
    pipeline.add_stage(
        DFPFileToDataFrameStage(config,
                                schema=source_schema,
                                file_type=FileTypes.JSON,
                                parser_kwargs={
                                    "lines": False, "orient": "records"
                                },
                                cache_dir=cache_dir))

    pipeline.add_stage(MonitorStage(config, description="Input data rate"))

    # This will split users or just use one single user
    pipeline.add_stage(
        DFPSplitUsersStage(config,
                           include_generic=include_generic,
                           include_individual=include_individual,
                           skip_users=skip_users,
                           only_users=only_users))

    # Next, have a stage that will create rolling windows
    pipeline.add_stage(
        DFPRollingWindowStage(
            config,
            min_history=300 if is_training else 1,
            min_increment=300 if is_training else 0,
            # For inference, we only ever want 1 day max
            max_history="60d" if is_training else "1d",
            cache_dir=cache_dir))

    # Output is UserMessageMeta -- Cached frame set
    pipeline.add_stage(DFPPreprocessingStage(config, input_schema=preprocess_schema, only_new_batches=not is_training))

    model_name_formatter = "DFP-duo-{user_id}"
    experiment_name_formatter = "dfp/duo/training/{reg_model_name}"

    if (is_training):

        # Finally, perform training which will output a model
        pipeline.add_stage(DFPTraining(config))

        pipeline.add_stage(MonitorStage(config, description="Training rate", smoothing=0.001))

        # Write that model to MLFlow
        pipeline.add_stage(
            DFPMLFlowModelWriterStage(config,
                                      model_name_formatter=model_name_formatter,
                                      experiment_name_formatter=experiment_name_formatter))
    else:
        pipeline.add_stage(DFPInferenceStage(config, model_name_formatter=model_name_formatter))

        pipeline.add_stage(MonitorStage(config, description="Inference rate", smoothing=0.001))

        pipeline.add_stage(DFPPostprocessingStage(config, z_score_threshold=2.0))

        pipeline.add_stage(WriteToFileStage(config, filename="dfp_detections_duo.csv", overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")