import logging
import os
from datetime import datetime
from datetime import timedelta

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
from dfp.utils.column_info import CustomColumn
from dfp.utils.column_info import DataFrameInputSchema
from dfp.utils.column_info import RenameColumn

import cudf

from morpheus._lib.file_types import FileTypes
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logging import configure_logging


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
    default="./.cache",
    show_envvar=True,
    help="The location to cache data such as S3 downloads and pre-processed data",
)
@click.option("--sample_rate_s",
              type=int,
              default=0,
              show_envvar=True,
              help="Minimum time step, in milliseconds, between object logs.")
@click.option(
    "--input_file",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="List of files to send to the visualization, in order",
)
@click.option('--tracking_uri',
              type=str,
              default="http://localhost:5000",
              help=("The ML Flow tracking URI to connect to the tracking backend. If not speficied, MF Flow will use "
                    "'file:///mlruns' relative to the current directory"))
def run_pipeline(train_users, skip_user, duration, cache_dir, sample_rate_s, **kwargs):

    source = "file"

    # To include the generic, we must be training all or generic
    include_generic = train_users == "all" or train_users == "generic"

    # To include individual, we must be either training or inferring
    include_individual = train_users != "generic"

    # None indicates we arent training anything
    is_training = train_users != "none"

    skip_users = list(skip_user)

    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

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
        'accessdevicebrowser', 'accessdeviceos', 'device', 'result', 'reason', 'logcount', "locincrement"
    ]
    config.ae.userid_column_name = "username"

    def column_logcount(df: cudf.DataFrame):
        per_day = df[config.ae.timestamp_column_name].dt.to_period("D")

        # Create the per-user, per-day log count
        return df.groupby([config.ae.userid_column_name, per_day]).cumcount()

    def column_locincrement(df: cudf.DataFrame):

        def per_user_calc(user_df: pd.DataFrame):
            try:
                city_column = "locationcity"
                state_column = None
                country_column = None

                pdf = user_df.copy()

                pdf['time'] = pd.to_datetime(pdf[config.ae.timestamp_column_name], errors='coerce')
                pdf['day'] = pdf['time'].dt.date

                pdf.sort_values(by=['time'], inplace=True)

                overall_location_columns = [
                    col for col in [city_column, state_column, country_column] if col is not None
                ]

                if len(overall_location_columns) > 0:
                    pdf["overall_location"] = pdf[city_column].str.cat([], sep="|", na_rep="-")

                    pdf['loc_cat'] = pdf.groupby('day')['overall_location'].transform(lambda x: pd.factorize(x)[0] + 1)

                    pdf.fillna({'loc_cat': 1}, inplace=True)

                    pdf['locincrement'] = pdf.groupby('day')['loc_cat'].expanding(1).max().droplevel(0)

                    pdf.drop(['overall_location', 'loc_cat'], inplace=True, axis=1)

                return pdf
            except Exception as ex:
                raise ex

        per_day = df[config.ae.timestamp_column_name].dt.to_period("D")

        # Simple but probably incorrect calculation
        return df.groupby([config.ae.userid_column_name, per_day, "locationcity"]).ngroup() + 1

        # # Create the per-user, per-day location increment count
        # # loc_cat = df.groupby([config.ae.userid_column_name,
        # #                       per_day])["locationcity"].transform(lambda x: pd.factorize(x)[0] + 1)
        # out_pdf = df.groupby(config.ae.userid_column_name).apply(lambda x: per_user_calc(x))

        # return out_pdf["locincrement"]

    def s3_date_extractor_duo(s3_object):
        key_object = s3_object.key

        # Extract the timestamp from the file name
        ts_object = key_object.split('_')[2].split('.json')[0].replace('T', ' ').replace('Z', '')
        ts_object = datetime.strptime(ts_object, '%Y-%m-%d %H:%M:%S.%f')

        return ts_object

    # Specify the column names to ensure all data is uniform
    column_info = [
        RenameColumn(name="accessdevicebrowser", dtype=str, input_name="access_device.browser"),
        RenameColumn(name="accessdeviceos", dtype=str, input_name="access_device.os"),
        RenameColumn(name="locationcity", dtype=str, input_name="auth_device.location.city"),
        RenameColumn(name="device", dtype=str, input_name="auth_device.name"),
        BoolColumn(name="result",
                   dtype=bool,
                   input_name="result",
                   true_values=["success", "SUCCESS"],
                   false_values=["denied", "DENIED", "FRAUD"]),
        RenameColumn(name="reason", dtype=str, input_name="reason"),
        RenameColumn(name="username", dtype=str, input_name="user.name"),
        RenameColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name=config.ae.timestamp_column_name),
    ]

    input_schema = DataFrameInputSchema(json_columns=["access_device", "application", "auth_device", "user"],
                                        column_info=column_info)

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
            DFPS3BatcherStage(config, sampling_rate_s=sample_rate_s, date_conversion_func=s3_date_extractor_duo))

        # Output is S3 Buckets. Convert to DataFrames. This caches downloaded S3 data
        pipeline.add_stage(
            DFPS3ToDataFrameStage(config,
                                  file_type=FileTypes.JSON,
                                  input_schema=input_schema,
                                  filter_null=False,
                                  s3_cache_dir=cache_dir))
    elif (source == "file"):
        pipeline.set_source(
            MultiFileSource(config,
                            input_schema=input_schema,
                            filenames=list(kwargs["input_file"]),
                            cudf_kwargs={
                                "lines": False, "orient": "records"
                            }))

    # This will split users or just use one single user
    pipeline.add_stage(
        DFPSplitUsersStage(config,
                           include_generic=include_generic,
                           include_individual=include_individual,
                           skip_users=skip_users))

    # Next, have a stage that will create rolling windows
    pipeline.add_stage(
        DFPRollingWindowStage(config, window_duration="1d", min_history=300, max_history=10000, s3_cache_dir=cache_dir))

    # Specify the final set of columns necessary just before pre-processing
    model_column_info = [
        # Input columns
        RenameColumn(name="accessdevicebrowser", dtype=str, input_name="accessdevicebrowser"),
        RenameColumn(name="accessdeviceos", dtype=str, input_name="accessdeviceos"),
        RenameColumn(name="device", dtype=str, input_name="device"),
        RenameColumn(name="result", dtype=bool, input_name="result"),
        RenameColumn(name="reason", dtype=str, input_name="reason"),
        # Derived columns
        CustomColumn(name="logcount", dtype=int, process_column_fn=column_logcount),
        CustomColumn(name="locincrement", dtype=int, process_column_fn=column_locincrement),
        # Extra columns
        RenameColumn(name="username", dtype=str, input_name="username"),
        RenameColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name=config.ae.timestamp_column_name),
    ]

    model_schema = DataFrameInputSchema(column_info=model_column_info)

    # Output is UserMessageMeta -- Cached frame set
    pipeline.add_stage(DFPPreprocessingStage(config, input_schema=model_schema))

    if (is_training):

        pipeline.add_stage(DFPTraining(config))

        pipeline.add_stage(DFPMLFlowModelWriterStage(config))
    else:
        pipeline.add_stage(DFPInferenceStage(config))

        pipeline.add_stage(DFPPostprocessingStage(config))

        if (source == "file"):
            pipeline.add_stage(WriteToFileStage(config, filename="dfp_detections.csv", overwrite=True))
        else:

            def print_s3_output(message: UserMessageMeta):
                print("WRITER MOCK: Writing the following: ")
                print(message.df)

            pipeline.add_stage(WriteToS3Stage(config, s3_writer=print_s3_output))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")
