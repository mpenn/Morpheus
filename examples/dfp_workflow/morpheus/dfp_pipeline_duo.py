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
from datetime import datetime
from datetime import timedelta
from functools import partial

import click
import mlflow
import pandas as pd
from dfp.stages.dfp_inference_stage import DFPInferenceStage
from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
from dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage
from dfp.stages.dfp_preprocessing_stage import DFPPreprocessingStage
from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage
from dfp.stages.dfp_s3_batcher_stage import DFPS3BatcherStage
from dfp.stages.dfp_s3_to_df import DFPS3ToDataFrameStage
from dfp.stages.dfp_split_users_stage import DFPSplitUsersStage
from dfp.stages.dfp_training import DFPTraining
from dfp.stages.multi_file_source import MultiFileSource
from dfp.stages.s3_object_source_stage import S3BucketSourceStage
from dfp.stages.s3_object_source_stage import s3_filter_duo
from dfp.stages.s3_object_source_stage import s3_object_generator
from dfp.stages.write_to_s3_stage import WriteToS3Stage
from dfp.utils.column_info import BoolColumn
from dfp.utils.column_info import ColumnInfo
from dfp.utils.column_info import CustomColumn
from dfp.utils.column_info import DataFrameInputSchema
from dfp.utils.column_info import DateTimeColumn
from dfp.utils.column_info import IncrementColumn
from dfp.utils.column_info import RenameColumn
from dfp.utils.column_info import column_listjoin
from dfp.utils.column_info import create_increment_col

from morpheus._lib.file_types import FileTypes
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.messages.message_meta import UserMessageMeta
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
    help="User IDs to skip",
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
@click.option(
    "--source",
    type=click.Choice(("file", "s3")),
    default="file",
    help="Data source",
)
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
    help="List of files to send to the visualization, in order",
)
@click.option('--tracking_uri',
              type=str,
              default="http://localhost:5000",
              help=("The ML Flow tracking URI to connect to the tracking backend. If not speficied, MF Flow will use "
                    "'file:///mlruns' relative to the current directory"))
def run_pipeline(train_users,
                 skip_user: typing.Tuple[str],
                 duration,
                 cache_dir,
                 log_level,
                 source,
                 sample_rate_s,
                 **kwargs):
    # To include the generic, we must be training all or generic
    include_generic = train_users == "all" or train_users == "generic"

    # To include individual, we must be either training or inferring
    include_individual = train_users != "generic"

    # None indicates we arent training anything
    is_training = train_users != "none"

    skip_users = list(skip_user)

    # Enable the Morpheus logger
    configure_logging(log_level=log_level)

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

    config.ae.feature_columns = [
        'accessdevicebrowser', 'accessdeviceos', 'device', 'result', 'reason', 'logcount', 'locincrement'
    ]
    config.ae.userid_column_name = "username"

    def s3_date_extractor_duo(s3_object):
        key_object = s3_object.key

        # Extract the timestamp from the file name
        ts_object = key_object.split('_')[2].split('.json')[0].replace('T', ' ').replace('Z', '')
        ts_object = datetime.strptime(ts_object, '%Y-%m-%d %H:%M:%S.%f')

        return ts_object

    # Specify the column names to ensure all data is uniform
    source_column_info = [
        RenameColumn(name="accessdevicebrowser", dtype=str, input_name="access_device.browser"),
        RenameColumn(name="accessdeviceos", dtype=str, input_name="access_device.os"),
        RenameColumn(name="locationcity", dtype=str, input_name="auth_device.location.city"),
        RenameColumn(name="device", dtype=str, input_name="auth_device.name"),
        BoolColumn(name="result",
                   dtype=bool,
                   input_name="result",
                   true_values=["success", "SUCCESS"],
                   false_values=["denied", "DENIED", "FRAUD"]),
        ColumnInfo(name="reason", dtype=str),
        RenameColumn(name="username", dtype=str, input_name="user.name"),
        DateTimeColumn(name="timestamp", dtype=datetime, input_name="timestamp"),
        CustomColumn(name="user.groups", dtype=str, process_column_fn=partial(column_listjoin, col_name="user.groups"))
    ]

    source_schema = DataFrameInputSchema(json_columns=["access_device", "application", "auth_device", "user"],
                                         column_info=source_column_info)

    # Preprocessing schema
    preprocess_column_info = [
        ColumnInfo(name="accessdevicebrowser", dtype=str),
        ColumnInfo(name="accessdeviceos", dtype=str),
        ColumnInfo(name="locationcity", dtype=str),
        ColumnInfo(name="device", dtype=str),
        ColumnInfo(name="result", dtype=bool),
        ColumnInfo(name="reason", dtype=str),
        ColumnInfo(name="username", dtype=str),
        ColumnInfo(name=config.ae.timestamp_column_name, dtype=datetime),
        # Derived columns
        IncrementColumn(name="logcount",
                        dtype=int,
                        input_name=config.ae.timestamp_column_name,
                        groupby_column="username"),
        CustomColumn(name="locincrement",
                     dtype=int,
                     process_column_fn=partial(create_increment_col, column_name="locationcity")),
    ]

    preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    if (source == "s3"):
        start_time = datetime.today() - timedelta(seconds=pd.Timedelta(duration).total_seconds())
        end_time = datetime.today()

        pipeline.set_source(
            S3BucketSourceStage(config,
                                object_generator=s3_object_generator(bucket_name="morpheus-duo-logs",
                                                                     filter_func=s3_filter_duo,
                                                                     start_date=start_time,
                                                                     end_date=end_time)))

        pipeline.add_stage(
            DFPS3BatcherStage(config,
                              period="D",
                              sampling_rate_s=sample_rate_s,
                              date_conversion_func=s3_date_extractor_duo))

        # Output is S3 Buckets. Convert to DataFrames. This caches downloaded S3 data
        pipeline.add_stage(
            DFPS3ToDataFrameStage(config,
                                  file_type=FileTypes.JSON,
                                  input_schema=source_schema,
                                  filter_null=False,
                                  cache_dir=cache_dir))

    elif (source == "file"):
        pipeline.set_source(
            MultiFileSource(config,
                            input_schema=source_schema,
                            filenames=list(kwargs["input_file"]),
                            parser_kwargs={
                                "lines": False, "orient": "records"
                            }))

    pipeline.add_stage(MonitorStage(config, description="Input data rate"))

    # This will split users or just use one single user
    pipeline.add_stage(
        DFPSplitUsersStage(config,
                           include_generic=include_generic,
                           include_individual=include_individual,
                           skip_users=skip_users))

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

    model_name_formatter = "AE-duo-{user_id}"
    experiment_name = "DFP-duo-training"

    if (is_training):

        pipeline.add_stage(DFPTraining(config))

        pipeline.add_stage(MonitorStage(config, description="Training rate", smoothing=0.001))

        pipeline.add_stage(
            DFPMLFlowModelWriterStage(config,
                                      model_name_formatter=model_name_formatter,
                                      experiment_name=experiment_name))
    else:
        pipeline.add_stage(DFPInferenceStage(config, model_name_formatter=model_name_formatter))

        pipeline.add_stage(MonitorStage(config, description="Inference rate", smoothing=0.001))

        pipeline.add_stage(DFPPostprocessingStage(config, z_score_threshold=3.0))

        if (source == "file"):
            pipeline.add_stage(WriteToFileStage(config, filename="dfp_detections.csv", overwrite=True))
        else:

            def print_s3_output(message: UserMessageMeta):
                logger.debug("WRITER MOCK: Writing the following: ")
                logger.debug(message.df)

            pipeline.add_stage(WriteToS3Stage(config, s3_writer=print_s3_output))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")
